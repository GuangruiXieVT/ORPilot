"""Google Gemini LLM provider."""

from __future__ import annotations

from pydantic import BaseModel

from .base import BaseLLM, ChatResponse, ToolCall, ToolDefinition


class GeminiLLM(BaseLLM):
    """Google Gemini provider via google-genai."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        max_tokens: int = 8192,
        timeout: float = 120.0,
        max_retries: int = 2,
        temperature: float = 0.0,
    ):
        from google import genai
        from google.genai import types as _gtypes

        client_kwargs: dict = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        # google-genai reads GOOGLE_API_KEY from env automatically when api_key is omitted
        self._client = genai.Client(**client_kwargs)
        self._gtypes = _gtypes
        self._model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._temperature = temperature
        self._timeout_exceptions = self._resolve_timeout_exc()
        self._rate_limit_exceptions = self._resolve_rate_limit_exc()
        super().__init__()

    @staticmethod
    def _resolve_timeout_exc() -> tuple:
        try:
            from google.api_core import exceptions as gex
            return (gex.DeadlineExceeded, gex.ServiceUnavailable, gex.InternalServerError)
        except ImportError:
            return (Exception,)

    @staticmethod
    def _resolve_rate_limit_exc() -> tuple:
        try:
            from google.api_core import exceptions as gex
            return (gex.ResourceExhausted,)
        except ImportError:
            return ()

    def _base_config(self, **extra) -> object:
        return self._gtypes.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            **extra,
        )

    # ------------------------------------------------------------------
    # JSON Schema → Gemini Schema conversion
    # ------------------------------------------------------------------

    def _resolve_refs(self, obj, defs: dict):
        """Recursively resolve $ref / anyOf / allOf for Gemini schema compat."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_name = obj["$ref"].split("/")[-1]
                return self._resolve_refs(defs.get(ref_name, {}), defs)
            if "anyOf" in obj:
                # Optional[X] → {"anyOf": [X, {"type": "null"}]}: take the non-null branch
                non_null = [s for s in obj["anyOf"] if s.get("type") != "null"]
                if len(non_null) == 1:
                    return self._resolve_refs(non_null[0], defs)
                return {"type": "object"}  # fallback for complex unions
            if "allOf" in obj:
                merged: dict = {}
                for sub in obj["allOf"]:
                    merged.update(self._resolve_refs(sub, defs))
                return merged
            return {k: self._resolve_refs(v, defs) for k, v in obj.items() if k not in ("$defs", "$schema")}
        if isinstance(obj, list):
            return [self._resolve_refs(item, defs) for item in obj]
        return obj

    def _flatten_schema(self, schema: dict) -> dict:
        """Inline all $defs/$ref so Gemini can consume the schema."""
        return self._resolve_refs(schema, schema.get("$defs", {}))

    def _json_schema_to_gemini(self, schema: dict):
        """Recursively convert a flattened JSON Schema dict to a Gemini Schema."""
        T = self._gtypes.Type
        type_map = {
            "string": T.STRING,
            "number": T.NUMBER,
            "integer": T.INTEGER,
            "boolean": T.BOOLEAN,
            "array": T.ARRAY,
            "object": T.OBJECT,
        }
        raw_type = schema.get("type", "object")
        if isinstance(raw_type, list):
            raw_type = next((t for t in raw_type if t != "null"), "object")
        schema_type = type_map.get(raw_type, T.OBJECT)

        kwargs: dict = {"type": schema_type}
        if schema.get("description"):
            kwargs["description"] = schema["description"]
        if schema.get("enum"):
            kwargs["enum"] = [str(e) for e in schema["enum"]]
        if schema.get("properties"):
            kwargs["properties"] = {
                k: self._json_schema_to_gemini(v)
                for k, v in schema["properties"].items()
            }
        if schema.get("required"):
            kwargs["required"] = schema["required"]
        if schema.get("items"):
            kwargs["items"] = self._json_schema_to_gemini(schema["items"])

        return self._gtypes.Schema(**kwargs)

    # ------------------------------------------------------------------
    # Message format: our standard dict list ↔ Gemini Content list
    # ------------------------------------------------------------------

    def _to_gemini_contents(self, messages: list[dict]) -> tuple[str | None, list]:
        """Convert internal messages to (system_instruction, gemini_contents).

        Messages appended by extend_messages carry a ``_gemini_content`` key
        (value is a native Gemini Content object) and are passed through as-is.
        """
        system = None
        contents = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
                continue
            if "_gemini_content" in m:
                contents.append(m["_gemini_content"])
                continue
            role = m.get("role", "user")
            gemini_role = "model" if role in ("assistant", "model") else "user"
            content = m.get("content", "")
            if isinstance(content, str):
                contents.append(self._gtypes.Content(
                    role=gemini_role,
                    parts=[self._gtypes.Part(text=content)],
                ))
        # Gemini requires the last turn to be from user
        if contents and getattr(contents[-1], "role", None) == "model":
            contents.append(self._gtypes.Content(
                role="user",
                parts=[self._gtypes.Part(text="Please proceed.")],
            ))
        return system, contents

    # ------------------------------------------------------------------
    # Token usage
    # ------------------------------------------------------------------

    def _capture_usage(self, response) -> None:
        meta = getattr(response, "usage_metadata", None)
        if meta:
            self._add_usage(
                getattr(meta, "prompt_token_count", 0) or 0,
                getattr(meta, "candidates_token_count", 0) or 0,
            )

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict]) -> str:
        messages = self._sanitize_messages(messages)
        system, contents = self._to_gemini_contents(messages)
        extra: dict = {}
        if system:
            extra["system_instruction"] = system

        def _call() -> str:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._base_config(**extra),
            )
            self._capture_usage(response)
            return response.text or ""

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)

    def structured_output(
        self, messages: list[dict], schema: type[BaseModel]
    ) -> BaseModel:
        """JSON-mode generation: force JSON output and validate with Pydantic."""
        messages = self._sanitize_messages(messages)
        system, contents = self._to_gemini_contents(messages)
        extra: dict = {"response_mime_type": "application/json"}
        if system:
            extra["system_instruction"] = system

        def _call() -> BaseModel:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._base_config(**extra),
            )
            self._capture_usage(response)
            raw = response.text or "{}"
            return schema.model_validate_json(self._strip_json_fences(raw))

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
    ) -> ChatResponse:
        messages = self._sanitize_messages(messages)
        system, contents = self._to_gemini_contents(messages)

        declarations = [
            self._gtypes.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=self._json_schema_to_gemini(self._flatten_schema(t.parameters)),
            )
            for t in tools
        ]
        extra: dict = {"tools": [self._gtypes.Tool(function_declarations=declarations)]}
        if system:
            extra["system_instruction"] = system

        def _call() -> ChatResponse:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._base_config(**extra),
            )
            self._capture_usage(response)

            candidate = response.candidates[0] if response.candidates else None
            if candidate is None:
                return ChatResponse(text="", tool_calls=[])

            text: str | None = None
            tool_calls: list[ToolCall] = []
            for idx, part in enumerate(candidate.content.parts):
                if part.text:
                    text = (text or "") + part.text
                elif part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        # Gemini has no native tool-call ID; use name+index for uniqueness
                        id=f"{fc.name}_{idx}",
                        name=fc.name,
                        arguments=dict(fc.args),
                    ))

            return ChatResponse(
                text=text,
                tool_calls=tool_calls,
                _payload=candidate.content,  # passed back via extend_messages
            )

        return self._retry(_call, self._max_retries, self._timeout_exceptions, self._rate_limit_exceptions)

    def extend_messages(
        self,
        messages: list[dict],
        response: ChatResponse,
        results: dict[str, str],
    ) -> list[dict]:
        new_messages = list(messages)
        # Model turn: the Content object containing the function_call parts
        new_messages.append({"_gemini_content": response._payload})
        # User turn: one FunctionResponse Part per tool call, in order
        result_parts = [
            self._gtypes.Part(
                function_response=self._gtypes.FunctionResponse(
                    name=tc.name,
                    response={"result": results.get(tc.id, "")},
                )
            )
            for tc in response.tool_calls
        ]
        new_messages.append({
            "_gemini_content": self._gtypes.Content(role="user", parts=result_parts)
        })
        return new_messages

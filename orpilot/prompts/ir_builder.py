"""System prompt for the IR builder LLM node."""

from ._loader import load_prompt

SYSTEM_PROMPT, _SYSTEM_VERSION = load_prompt("ir_system.md")

"""System prompt for the TextIngestor LLM call."""

from ._loader import load_prompt

SYSTEM_PROMPT, _SYSTEM_VERSION = load_prompt("ingestor_system.md")

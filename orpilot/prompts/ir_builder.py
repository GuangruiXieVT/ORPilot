"""System prompt for the IR builder LLM node."""

from pathlib import Path

SYSTEM_PROMPT = (Path(__file__).parent / "ir_prompt.txt").read_text()

"""Prompts for translating solutions to natural language reports."""

from ._loader import load_prompt

SYSTEM_PROMPT, _SYSTEM_VERSION = load_prompt("report_system.md")

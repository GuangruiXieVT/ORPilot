"""Prompts for the interview / need-elicitation stage."""

from ._loader import load_prompt

SYSTEM_PROMPT, _SYSTEM_VERSION = load_prompt("interview_system.md")
SUMMARIZE_PROMPT, _SUMMARIZE_VERSION = load_prompt("interview_summarize.md")

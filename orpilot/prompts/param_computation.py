"""Prompt for the parameter computation agent."""

from ._loader import load_prompt

SYSTEM_PROMPT, _SYSTEM_VERSION = load_prompt("param_computation_system.md")
USER_PROMPT_TEMPLATE, _USER_VERSION = load_prompt("param_computation_user.md")

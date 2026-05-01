"""Prompts for guiding CSV-based data collection from the user."""

from ._loader import load_prompt

SYSTEM_PROMPT, _SYSTEM_VERSION = load_prompt("data_guide_system.md")
SPEC_EXTRACTION_PROMPT, _SPEC_VERSION = load_prompt("data_guide_spec_extraction.md")

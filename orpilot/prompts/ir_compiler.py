"""Prompts for the IR compiler LLM retry path."""

RETRY_PROMPT = """\
The following solver code failed with this error:
{error}

Previous code:
```python
{previous_code}
```
Fix the code. Output ONLY the corrected Python code, no markdown, no explanation.
"""

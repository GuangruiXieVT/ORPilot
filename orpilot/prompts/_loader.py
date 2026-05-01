"""Prompt file loader with YAML front-matter version extraction.

Each prompt file may start with a YAML front-matter block:

    ---
    version: 1.2.0
    ---

    <prompt content>

The loader strips the front-matter before returning the content, so the
version tag never reaches the LLM.  Files without front-matter are returned
as-is with version ``"unversioned"``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(filename: str) -> tuple[str, str]:
    """Return ``(content, version)`` for a prompt file in the prompts directory.

    Front-matter is stripped from the returned content.
    """
    return _load(filename)


@lru_cache(maxsize=None)
def _load(filename: str) -> tuple[str, str]:
    raw = (_PROMPTS_DIR / filename).read_text(encoding="utf-8")

    if raw.startswith("---\n"):
        try:
            end = raw.index("\n---\n", 4)
        except ValueError:
            return raw.strip(), "unversioned"
        frontmatter = raw[4:end]
        content = raw[end + 5:].strip()
        version = "unversioned"
        for line in frontmatter.splitlines():
            if line.startswith("version:"):
                version = line.split(":", 1)[1].strip()
                break
        return content, version

    return raw.strip(), "unversioned"


def all_versions() -> dict[str, str]:
    """Return a mapping of prompt filename → version for every .md prompt file."""
    versions: dict[str, str] = {}
    for path in sorted(_PROMPTS_DIR.glob("*.md")):
        _, version = _load(path.name)
        versions[path.name] = version
    return versions

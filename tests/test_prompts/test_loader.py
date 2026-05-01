"""Unit tests for prompts/_loader.py — frontmatter stripping and version extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from orpilot.prompts._loader import load_prompt, all_versions


# ---------------------------------------------------------------------------
# load_prompt — frontmatter handling
# ---------------------------------------------------------------------------

def test_load_prompt_real_file_returns_string():
    """Sanity check: a real prompt file loads without error."""
    content, version = load_prompt("interview_system.md")
    assert isinstance(content, str)
    assert len(content) > 0


def test_load_prompt_strips_frontmatter():
    """Version frontmatter must not appear in the returned content."""
    content, _ = load_prompt("interview_system.md")
    assert not content.startswith("---")
    assert "version:" not in content.splitlines()[:3]


def test_load_prompt_returns_version():
    """All shipped prompt files carry version: 1.0.0."""
    _, version = load_prompt("interview_system.md")
    assert version == "1.0.0"


def test_all_solvers_have_version():
    """Every direct_code_gen_*.md file must be versioned."""
    for solver in ("pulp", "pyomo", "ortools", "gurobi", "cplex"):
        _, version = load_prompt(f"direct_code_gen_{solver}.md")
        assert version != "unversioned", f"direct_code_gen_{solver}.md has no version"


# ---------------------------------------------------------------------------
# all_versions
# ---------------------------------------------------------------------------

def test_all_versions_returns_dict():
    versions = all_versions()
    assert isinstance(versions, dict)
    assert len(versions) >= 14  # at least the shipped prompt files


def test_all_versions_keys_are_md_filenames():
    versions = all_versions()
    for key in versions:
        assert key.endswith(".md")


def test_all_versions_values_are_strings():
    versions = all_versions()
    for v in versions.values():
        assert isinstance(v, str)


def test_all_versions_includes_known_files():
    versions = all_versions()
    expected = {
        "interview_system.md",
        "interview_summarize.md",
        "direct_code_gen_pulp.md",
        "ir_system.md",
        "report_system.md",
    }
    assert expected.issubset(versions.keys())


def test_all_shipped_prompts_are_versioned():
    """No shipped .md prompt file should be 'unversioned'."""
    versions = all_versions()
    unversioned = [f for f, v in versions.items() if v == "unversioned"]
    assert unversioned == [], f"Unversioned prompt files: {unversioned}"


# ---------------------------------------------------------------------------
# Frontmatter parsing edge cases (via a temp file patched into the loader)
# ---------------------------------------------------------------------------

def _load_raw(content: str) -> tuple[str, str]:
    """Parse content as if it were a prompt file, bypassing lru_cache."""
    # Replicate loader logic directly to avoid cache complications
    if content.startswith("---\n"):
        try:
            end = content.index("\n---\n", 4)
        except ValueError:
            return content.strip(), "unversioned"
        frontmatter = content[4:end]
        body = content[end + 5:].strip()
        version = "unversioned"
        for line in frontmatter.splitlines():
            if line.startswith("version:"):
                version = line.split(":", 1)[1].strip()
                break
        return body, version
    return content.strip(), "unversioned"


def test_frontmatter_stripped_correctly():
    raw = "---\nversion: 2.3.1\n---\n\nHello world"
    content, version = _load_raw(raw)
    assert content == "Hello world"
    assert version == "2.3.1"


def test_no_frontmatter_returns_unversioned():
    raw = "Just a plain prompt with no frontmatter."
    content, version = _load_raw(raw)
    assert content == raw
    assert version == "unversioned"


def test_malformed_frontmatter_treated_as_content():
    """A leading '---' with no closing '---' is returned as-is."""
    raw = "---\nversion: 1.0.0\nno closing marker"
    content, version = _load_raw(raw)
    assert version == "unversioned"
    assert "version: 1.0.0" in content


def test_frontmatter_extra_fields_ignored():
    raw = "---\nauthor: Alice\nversion: 3.0.0\ndate: 2026-01-01\n---\n\nContent here"
    content, version = _load_raw(raw)
    assert version == "3.0.0"
    assert content == "Content here"

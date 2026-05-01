"""Project-level config file discovery and loading."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

_AUTO_CONFIG_NAMES = ("orpilot.toml", "orpilot.json")


def load_config_file(path: Path) -> dict:
    """Load config values from a TOML or JSON file."""
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with open(path, "rb") as f:
        return tomllib.load(f)


def discover_config_file() -> Path | None:
    """Locate a config file via ORPILOT_CONFIG env var or by walking up from CWD."""
    env_path = os.environ.get("ORPILOT_CONFIG")
    if env_path:
        return Path(env_path)

    current = Path.cwd()
    while True:
        for name in _AUTO_CONFIG_NAMES:
            candidate = current / name
            if candidate.exists():
                return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def load_project_config() -> dict:
    """Discover and load the project config file; return empty dict if none found."""
    path = discover_config_file()
    if path is None or not path.exists():
        return {}
    return load_config_file(path)

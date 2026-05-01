"""Pytest fixtures for benchmark tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--save-dir",
        default=None,
        help="Directory to save generated IR, code, and LP files for each benchmark case.",
    )
    parser.addoption(
        "--generate-ir",
        action="store_true",
        default=None,
        help="Generate an IR blueprint after each successful solve. Falls back to generate_ir in orpilot.toml.",
    )
    parser.addoption(
        "--difficulty",
        default=None,
        help="Filter IndustryOR cases by difficulty: Easy|Medium|Hard (default: Easy).",
    )
    parser.addoption(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of dataset cases to run (default: all).",
    )
    parser.addoption(
        "--start",
        type=int,
        default=0,
        help="Row index to start from (0-based). E.g. --start 22 skips the first 22 cases.",
    )
    parser.addoption(
        "--temperature",
        type=float,
        default=None,
        help="LLM sampling temperature (0.0 = deterministic). Falls back to temperature in orpilot.toml.",
    )
    parser.addoption(
        "--solver",
        default=None,
        help="OR solver backend: pulp, pyomo, ortools, gurobi, cplex. Falls back to solver in orpilot.toml, then 'pulp'.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: benchmark tests (no LLM required)")
    config.addinivalue_line("markers", "llm: benchmark tests that require a live LLM API key")
    config.addinivalue_line("markers", "industryOR: tests against CardinalOperations/IndustryOR dataset")
    config.addinivalue_line("markers", "NL4OPT: tests against CardinalOperations/NL4OPT dataset")
    config.addinivalue_line("markers", "NLP4LP: tests against udell-lab/NLP4LP dataset")


@pytest.fixture(scope="session")
def generate_ir(request) -> bool:
    """Return True if on-demand IR generation after successful solve is enabled.

    Precedence: --generate-ir CLI flag > generate_ir in orpilot.toml > False.
    """
    from orpilot.config import load_project_config

    cli_flag = request.config.getoption("--generate-ir")
    if cli_flag:
        return True
    cfg = load_project_config()
    return bool(cfg.get("generate_ir", False))


@pytest.fixture(scope="session")
def difficulty(request) -> str:
    """Return the --difficulty value (default: Easy)."""
    return request.config.getoption("--difficulty") or "Easy"


@pytest.fixture(scope="session")
def limit(request) -> int | None:
    """Return the --limit value, or None to run all cases."""
    return request.config.getoption("--limit")


@pytest.fixture(scope="session")
def start(request) -> int:
    """Return the --start value (0-based row offset, default 0)."""
    return request.config.getoption("--start") or 0


@pytest.fixture(scope="session")
def temperature(request) -> float:
    """Return the LLM sampling temperature.

    Precedence: --temperature CLI flag > temperature in orpilot.toml > 0.0.
    """
    from orpilot.config import load_project_config

    cli_val = request.config.getoption("--temperature")
    if cli_val is not None:
        return cli_val
    cfg = load_project_config()
    return float(cfg.get("temperature", 0.0))




@pytest.fixture(scope="session")
def solver(request) -> str:
    """Return the OR solver backend to use.

    Precedence: --solver CLI flag > solver in orpilot.toml > 'pulp'.
    """
    from orpilot.config import load_project_config

    cli_val = request.config.getoption("--solver")
    if cli_val is not None:
        return cli_val
    cfg = load_project_config()
    return cfg.get("solver", "pulp")


@pytest.fixture(scope="session")
def save_dir(request) -> Path | None:
    """Return the --save-dir path (created on first use), or None if not set."""
    raw = request.config.getoption("--save-dir")
    if raw is None:
        return None
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(scope="session")
def llm_fixture(temperature):
    """Return an LLM instance configured from orpilot.toml / env vars, else skip."""
    from orpilot.config import load_project_config
    from orpilot.llm.config import LLMConfig, get_llm

    cfg = load_project_config()

    provider = os.getenv("ORPILOT_LLM_PROVIDER") or cfg.get("provider")
    model    = os.getenv("ORPILOT_MODEL")         or cfg.get("model")
    base_url = os.getenv("OPENAI_BASE_URL")        or cfg.get("base_url")
    api_key  = (
        cfg.get("api_key")
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    # Infer provider from key if not explicitly set
    if not provider:
        if os.getenv("ANTHROPIC_API_KEY") or (api_key and cfg.get("provider") == "anthropic"):
            provider = "anthropic"
        else:
            provider = "openai"

    if not api_key:
        pytest.skip("No LLM API key available (set ANTHROPIC_API_KEY, OPENAI_API_KEY, or api_key in orpilot.toml)")

    return get_llm(LLMConfig(provider=provider, model=model, api_key=api_key, base_url=base_url, temperature=temperature))

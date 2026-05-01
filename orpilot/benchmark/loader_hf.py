"""HuggingFace dataset loader for benchmark cases."""

from __future__ import annotations


def load_hf_cases(
    dataset: str,
    split: str = "test",
    difficulty: str | None = None,
    ids: list[int] | None = None,
    limit: int | None = None,
) -> list:
    """Load BenchmarkCase objects from a HuggingFace dataset.

    Parameters
    ----------
    dataset:
        HuggingFace dataset name, e.g. ``"CardinalOperations/IndustryOR"``.
    split:
        Dataset split to load (default ``"test"``).
    difficulty:
        Optional filter: ``"Easy"``, ``"Medium"``, or ``"Hard"`` (case-insensitive).
    ids:
        Optional list of row ``id`` values to include.
    limit:
        Maximum number of cases to return.

    Returns
    -------
    list[BenchmarkCase]
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace dataset loading. "
            "Install it with: pip install 'orpilot[hf]'  or  pip install datasets"
        ) from exc

    from orpilot.benchmark.case import BenchmarkCase

    hf_data = load_dataset(dataset, split=split)

    cases: list[BenchmarkCase] = []
    for row in hf_data:
        # Filter by ids
        if ids is not None and row["id"] not in ids:
            continue

        # Filter by difficulty (case-insensitive)
        if difficulty is not None and row["difficulty"].lower() != difficulty.lower():
            continue

        expected = float(row["en_answer"])
        tol = max(1e-4, abs(expected) * 1e-3)

        case = BenchmarkCase(
            name=f"industryOR_{row['id']:03d}",
            problem_text=row["en_question"],
            tables=None,
            ir_model=None,
            expected_objective=expected,
            objective_tolerance=tol,
            source=f"{dataset}/{split}",
            tags=[row["difficulty"].lower(), "industryOR"],
        )
        cases.append(case)

        if limit is not None and len(cases) >= limit:
            break

    return cases


# ---------------------------------------------------------------------------
# Column-name candidates for NLP4LP (gated, schema may vary by data_cleaner
# version).  We try each in order and use the first key that exists in a row.
# ---------------------------------------------------------------------------
_NLP4LP_TEXT_FIELDS = (
    "description", "problem_description", "problem_text", "text", "question",
)
_NLP4LP_ANSWER_FIELDS = (
    "answer", "optimal_value", "obj_value", "objective", "opt",
    "optimal", "obj", "val", "value", "sol", "result",
)


def _resolve_text(row: dict) -> str | None:
    for key in _NLP4LP_TEXT_FIELDS:
        if key in row and row[key]:
            return str(row[key])
    return None


def _resolve_answer(row: dict) -> float | None:
    """Return the optimal objective value from a row, or None to skip the row."""
    # Direct numeric/string field at the top level
    for key in _NLP4LP_ANSWER_FIELDS:
        if key in row:
            val = row[key]
            if val is None:
                return None
            # solution dict stored directly under a known answer key
            if isinstance(val, dict):
                for sub in _NLP4LP_ANSWER_FIELDS:
                    if sub in val:
                        try:
                            return float(val[sub])
                        except (TypeError, ValueError):
                            return None
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None  # non-numeric → infeasible/unsolved, skip

    # 'solution' may be a JSON string, a dict, or a bare scalar.
    import json as _json

    sol_raw = row.get("solution")
    if sol_raw is not None:
        # Parse JSON string → dict/scalar
        if isinstance(sol_raw, str):
            try:
                sol_raw = _json.loads(sol_raw)
            except (ValueError, TypeError):
                try:
                    return float(sol_raw)
                except (ValueError, TypeError):
                    return None

        if isinstance(sol_raw, dict):
            # Try known answer keys at the top level of the solution dict
            for key in _NLP4LP_ANSWER_FIELDS:
                if key in sol_raw:
                    try:
                        return float(sol_raw[key])
                    except (TypeError, ValueError):
                        return None
            # NLP4LP stores {"variables": {...}, "objective": X} — try one level deeper
            for sub_val in sol_raw.values():
                if isinstance(sub_val, (int, float)):
                    # bare numeric sibling of "variables" is likely the objective
                    try:
                        return float(sub_val)
                    except (TypeError, ValueError):
                        pass
        else:
            try:
                return float(sol_raw)
            except (TypeError, ValueError):
                return None

    return None


def _nlp4lp_schema_error(first_row: dict) -> str:
    """Return a diagnostic message showing the actual row structure."""
    import json as _json

    # Summarise each field: show type and a short preview of the value.
    summary = {}
    for k, v in first_row.items():
        if isinstance(v, str):
            summary[k] = f"str: {v[:80]!r}"
        elif isinstance(v, dict):
            summary[k] = f"dict with keys {list(v.keys())}"
        else:
            summary[k] = f"{type(v).__name__}: {v!r}"

    return (
        "Could not resolve the answer field from NLP4LP rows.\n"
        f"First-row columns and values:\n{_json.dumps(summary, indent=2)}\n"
        f"Tried text fields: {_NLP4LP_TEXT_FIELDS}\n"
        f"Tried answer fields: {_NLP4LP_ANSWER_FIELDS}"
    )


def load_nl4opt_cases(
    dataset: str = "CardinalOperations/NL4OPT",
    split: str = "test",
    offset: int = 0,
    limit: int | None = None,
) -> list:
    """Load BenchmarkCase objects from the NL4OPT HuggingFace dataset.

    The NL4OPT dataset has only ``en_question`` and ``en_answer`` columns
    (no ``id`` or ``difficulty`` fields).

    Parameters
    ----------
    dataset:
        HuggingFace dataset name (default ``"CardinalOperations/NL4OPT"``).
    split:
        Dataset split to load (default ``"test"``).
    offset:
        Number of rows to skip from the start (default 0).
    limit:
        Maximum number of cases to return.

    Returns
    -------
    list[BenchmarkCase]
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace dataset loading. "
            "Install it with: pip install 'orpilot[hf]'  or  pip install datasets"
        ) from exc

    from orpilot.benchmark.case import BenchmarkCase

    hf_data = load_dataset(dataset, split=split)

    cases: list[BenchmarkCase] = []
    for i, row in enumerate(hf_data):
        if i < offset:
            continue
        raw_answer = row["en_answer"]
        try:
            expected = float(raw_answer)
            tol = max(1e-4, abs(expected) * 1e-3)
            expected_status = "optimal"
        except (TypeError, ValueError):
            # Rows like 'No Best Solution' are infeasible/unbounded — skip them.
            continue

        case = BenchmarkCase(
            name=f"nl4opt_{i:03d}",
            problem_text=row["en_question"],
            tables=None,
            ir_model=None,
            expected_objective=expected,
            objective_tolerance=tol,
            expected_status=expected_status,
            source=f"{dataset}/{split}",
            tags=["nl4opt"],
        )
        cases.append(case)

        if limit is not None and len(cases) >= limit:
            break

    return cases


def load_nlp4lp_cases(
    dataset: str = "udell-lab/NLP4LP",
    split: str = "test",
    token: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list:
    """Load BenchmarkCase objects from the NLP4LP HuggingFace dataset.

    NLP4LP is a **gated** dataset — callers must supply a HuggingFace token
    via *token* or the ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` environment
    variable.  Rows that are infeasible, unsolved, or have a non-numeric
    answer are skipped automatically.

    The loader tries a prioritised list of column-name candidates for both the
    problem text and the optimal value so that it remains functional across
    schema versions produced by different ``data_cleaner.py`` revisions.

    Parameters
    ----------
    dataset:
        HuggingFace dataset name (default ``"udell-lab/NLP4LP"``).
    split:
        Dataset split to load (default ``"test"``).
    token:
        HuggingFace access token.  Falls back to the ``HF_TOKEN`` /
        ``HUGGING_FACE_HUB_TOKEN`` environment variables.
    offset:
        Number of rows to skip from the start (default 0).
    limit:
        Maximum number of cases to return.

    Returns
    -------
    list[BenchmarkCase]
    """
    import os

    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace dataset loading. "
            "Install it with: pip install 'orpilot[hf]'  or  pip install datasets"
        ) from exc

    from orpilot.benchmark.case import BenchmarkCase

    hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "NLP4LP is a gated dataset. Provide a HuggingFace token via the "
            "HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable, or pass "
            "token=... to load_nlp4lp_cases()."
        )

    hf_data = load_dataset(dataset, split=split, token=hf_token)

    # Validate columns against the first row so we fail early with a clear message.
    if hf_data:
        first = dict(hf_data[0])
        if _resolve_text(first) is None:
            raise ValueError(
                f"Could not find a problem-text field in NLP4LP rows. "
                f"Available columns: {list(first.keys())}. "
                f"Expected one of: {_NLP4LP_TEXT_FIELDS}"
            )

    cases: list[BenchmarkCase] = []
    first_row: dict | None = None
    for i, row in enumerate(hf_data):
        if first_row is None:
            first_row = dict(row)
        if i < offset:
            continue

        problem_text = _resolve_text(row)
        if not problem_text:
            continue

        # Append parsed parameters so the LLM receives concrete numerical data.
        params_raw = row.get("parameters")
        if params_raw:
            import json as _json
            try:
                params = _json.loads(params_raw) if isinstance(params_raw, str) else params_raw
                problem_text = (
                    problem_text.rstrip()
                    + "\n\nParameters:\n"
                    + _json.dumps(params, indent=2)
                )
            except (ValueError, TypeError):
                problem_text = problem_text.rstrip() + "\n\nParameters:\n" + str(params_raw)

        expected = _resolve_answer(row)
        if expected is None:
            # Infeasible, unsolved, or non-numeric answer — skip.
            continue

        tol = max(1e-4, abs(expected) * 1e-3)
        case = BenchmarkCase(
            name=f"nlp4lp_{i:03d}",
            problem_text=problem_text,
            tables=None,
            ir_model=None,
            expected_objective=expected,
            objective_tolerance=tol,
            expected_status="optimal",
            source=f"{dataset}/{split}",
            tags=["nlp4lp"],
        )
        cases.append(case)

        if limit is not None and len(cases) >= limit:
            break

    if not cases and first_row is not None:
        raise ValueError(_nlp4lp_schema_error(first_row))

    return cases


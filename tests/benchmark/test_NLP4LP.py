"""NLP4LP HuggingFace dataset benchmark tests — require a live LLM API key and HF token."""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from pathlib import Path

import pytest

from orpilot.benchmark.runner import BenchmarkRunner

log = logging.getLogger(__name__)


def _save_artifacts(result, case, out_dir: Path) -> None:
    """Write ir.json, model.py, and model.lp for a case into out_dir/<case_name>/."""
    case_dir = out_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    if result.ir_model:
        (case_dir / "ir.json").write_text(json.dumps(result.ir_model, indent=2), encoding="utf-8")
    if result.generated_code:
        (case_dir / "model.py").write_text(result.generated_code, encoding="utf-8")
    if result.lp_content:
        (case_dir / "model.lp").write_text(result.lp_content, encoding="utf-8")


@pytest.mark.llm
@pytest.mark.NLP4LP
def test_NLP4LP_sample(llm_fixture, save_dir, generate_ir, start, limit, solver):
    """Load NLP4LP cases and run the direct code gen pipeline for each.

    NLP4LP is a gated HuggingFace dataset — set HF_TOKEN or
    HUGGING_FACE_HUB_TOKEN in your environment before running.

    Pass --limit N to cap the number of cases (default: all).
    Pass --generate-ir to also produce a solver-agnostic IR blueprint after
    each successful solve (or set generate_ir=true in orpilot.toml).
    """
    pytest.importorskip("datasets", reason="pip install 'orpilot[hf]' to run NLP4LP tests")

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        pytest.skip(
            "NLP4LP is a gated dataset. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN "
            "to run this benchmark."
        )

    from orpilot.benchmark.loader_hf import load_nlp4lp_cases

    cases = load_nlp4lp_cases(token=hf_token, offset=start, limit=limit)
    assert cases, "No solvable cases loaded from udell-lab/NLP4LP (all rows were infeasible/non-numeric)"

    runner = BenchmarkRunner(timeout=180)
    passed = 0
    failures: list[str] = []
    solve_times: list[float] = []
    results: list = []

    mode_label = "direct+ir" if generate_ir else "direct"
    log.info("Running %d NLP4LP cases (mode=%s, limit=%s)", len(cases), mode_label, limit)
    suite_start = time.monotonic()
    for i, case in enumerate(cases, 1):
        log.info("[%d/%d] %s (expected %s) ...", i, len(cases), case.name, case.expected_objective)

        if hasattr(llm_fixture, "reset_usage"):
            llm_fixture.reset_usage()

        result = runner.run_direct_pipeline(case, llm_fixture, solver=solver, generate_ir=generate_ir)

        if hasattr(llm_fixture, "get_usage"):
            usage = llm_fixture.get_usage()
            result.metrics = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "latency_s": round(result.solve_time or 0.0, 2),
            }

        results.append(result)
        solve_times.append(result.solve_time)
        if save_dir:
            _save_artifacts(result, case, save_dir)
        if result.passed:
            passed += 1
            log.info("  PASS  (obj=%s, time=%.1fs)", result.objective_value, result.solve_time)
        else:
            msg = (
                f"status={result.status}, "
                f"obj={result.objective_value} (expected {case.expected_objective}), "
                f"error={result.error}"
            )
            log.info("  FAIL  (%s)", msg)
            failures.append(f"{case.name}: {msg}")
        print(f"[{i}/{len(cases)}] {'PASS' if result.passed else 'FAIL'}  passed={passed}  failed={len(failures)}  ({result.solve_time:.1f}s)", flush=True)

    total = len(cases)
    pct = 100.0 * passed / total if total else 0.0
    total_wall = time.monotonic() - suite_start
    avg_t = sum(solve_times) / len(solve_times) if solve_times else 0.0
    min_t = min(solve_times) if solve_times else 0.0
    max_t = max(solve_times) if solve_times else 0.0
    log.info("=" * 60)
    log.info("  RESULT: %d/%d passed (%.1f%%)", passed, total, pct)
    log.info("  TIME:   total=%.1fs  avg=%.1fs  min=%.1fs  max=%.1fs", total_wall, avg_t, min_t, max_t)
    if failures:
        log.info("  FAILED cases:")
        for f in failures:
            log.info("    - %s", f)
    log.info("=" * 60)

    if save_dir and results:
        total_input = sum(r.metrics.get("input_tokens", 0) for r in results)
        total_output = sum(r.metrics.get("output_tokens", 0) for r in results)
        aggregate = {
            "run_id": datetime.datetime.now().isoformat(timespec="seconds"),
            "solver": solver,
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "totals": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "latency_s": round(total_wall, 2),
            },
            "cases": {
                r.case_name: {
                    "status": r.status,
                    "passed": r.passed,
                    "input_tokens": r.metrics.get("input_tokens", 0),
                    "output_tokens": r.metrics.get("output_tokens", 0),
                    "latency_s": r.metrics.get("latency_s", 0.0),
                }
                for r in results
            },
        }
        (save_dir / "metrics.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        log.info("  Wrote aggregate metrics to %s/metrics.json", save_dir)

    assert not failures, f"{len(failures)}/{total} NLP4LP cases failed:\n" + "\n".join(failures)

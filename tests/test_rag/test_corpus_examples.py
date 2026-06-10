"""Test that every corpus example compiles and solves to the expected objective."""
import csv
import json
import os
import re
import pytest

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "../../corpus/examples")


def load_corpus_examples():
    """Return list of (name, ir_dict, data_dict, expected_obj) tuples."""
    examples = []
    for fname in sorted(os.listdir(CORPUS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(CORPUS_DIR, fname)
        with open(path) as f:
            ex = json.load(f)

        ir = ex.get("ir")
        if not ir:
            continue

        # Load CSV data from sibling _data/ directory
        stem = fname[:-5]  # strip .json
        data_dir = os.path.join(CORPUS_DIR, stem + "_data")
        data = {}
        if os.path.isdir(data_dir):
            for csv_file in os.listdir(data_dir):
                if csv_file.endswith(".csv"):
                    table = csv_file[:-4]
                    with open(os.path.join(data_dir, csv_file)) as f:
                        data[table] = list(csv.DictReader(f))

        # Extract expected objective from problem metadata if present
        expected_obj = ex.get("problem", {}).get("expected_objective")

        examples.append((stem, ir, data, expected_obj))
    return examples


def _round_obj(val, tol=1e-3):
    """Round objective to 4 sig figs for comparison."""
    if val is None:
        return None
    if abs(val) < 1e-9:
        return 0.0
    magnitude = 10 ** (len(str(int(abs(val)))) - 1)
    return round(val / magnitude, 4) * magnitude


# Build parametrized test cases at collection time
_EXAMPLES = load_corpus_examples()


@pytest.mark.parametrize("name,ir,data,expected_obj", _EXAMPLES, ids=[e[0] for e in _EXAMPLES])
def test_corpus_example_solves(name, ir, data, expected_obj):
    """Each corpus example must compile, solve to 'optimal' or 'feasible', and hit expected obj."""
    from orpilot.codegen.ir_compiler import IRCompiler

    # Compile
    compiler = IRCompiler()
    try:
        code = compiler.compile(ir)
    except Exception as e:
        pytest.fail(f"{name}: compile failed — {e}")

    # Execute
    g = {}
    exec(code, g)  # noqa: S102
    try:
        result = g["solve"](data)
    except Exception as e:
        pytest.fail(f"{name}: solve() raised — {e}")

    status = result.get("status")
    assert status in ("optimal", "feasible"), f"{name}: bad status '{status}'"

    obj = result.get("objective_value")
    assert obj is not None, f"{name}: objective_value is None"

    # If a known expected objective is embedded in the JSON, check within 0.5%
    if expected_obj is not None:
        rel_err = abs(obj - expected_obj) / (abs(expected_obj) + 1e-9)
        assert rel_err < 0.005, (
            f"{name}: obj={obj:.6g} expected={expected_obj:.6g} (rel_err={rel_err:.4%})"
        )

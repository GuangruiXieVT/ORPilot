"""Safe execution of generated solver code in a subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


class CodeExecutor:
    """Execute generated Python solver code in a restricted subprocess."""

    def __init__(self, timeout: int = 120, allowed_modules: list[str] | None = None):
        self.timeout = timeout
        self.allowed_modules = allowed_modules or []

    def execute(self, code: str, data: dict, time_limit: int | None = None, show_solver_log: bool = False) -> dict:
        """Run the generated code's `solve(data)` function in a subprocess.

        Args:
            code: Generated solver code containing a ``solve(data)`` function.
            data: Data dictionary passed to the solve function.
            time_limit: Optional solver time limit in seconds.  The wrapper
                forwards this to ``solve(data, time_limit=...)``.  The
                subprocess kill-timeout is set to ``time_limit + 30`` so the
                solver has time to return its best incumbent before the process
                is forcibly killed.

        Returns a dict with keys:
            - result: the dict returned by solve(data), or None
            - stdout: captured stdout
            - error: error message string, or None
        """
        code = self._strip_code_fences(code)
        wrapper = self._build_wrapper(code, time_limit=time_limit, show_solver_log=show_solver_log)

        # Give the subprocess a hard ceiling beyond the solver limit so it can
        # finish gracefully (serialize results, write LP, etc.).  The buffer is
        # 20 % of the solver limit, with a minimum of 60 s and a maximum of
        # 120 s.  Falls back to self.timeout when no solver limit is set.
        if time_limit is not None:
            buffer = max(30, min(60, int(time_limit * 0.2)))
            proc_timeout = time_limit + buffer
        else:
            proc_timeout = self.timeout

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            script_path = tmpdir_path / "solver_script.py"
            script_path.write_text(wrapper)

            # Write data to a separate file so it is never embedded in the script.
            # The subprocess loads it from disk, keeping memory usage proportional
            # to the data size rather than doubling it as a string literal.
            data_path = tmpdir_path / "data.json"
            data_path.write_text(json.dumps(data), encoding="utf-8")

            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=proc_timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return {
                    "result": None,
                    "stdout": "",
                    "error": f"Execution timed out after {proc_timeout}s",
                }

            if proc.returncode != 0:
                return {
                    "result": None,
                    "stdout": proc.stdout,
                    "error": proc.stderr or f"Process exited with code {proc.returncode}",
                    "lp_content": self._read_lp_file(tmpdir),
                }

            # Parse the JSON result from stdout
            stdout_lines = proc.stdout.strip().split("\n")
            # The last line should be our JSON marker
            result = None
            output_lines = []
            for line in stdout_lines:
                if line.startswith("__ORPILOT_RESULT__:"):
                    try:
                        result = json.loads(line[len("__ORPILOT_RESULT__:"):])
                    except json.JSONDecodeError:
                        pass
                else:
                    output_lines.append(line)

            if show_solver_log and output_lines:
                print("\n".join(output_lines), flush=True)

            if result is None:
                return {
                    "result": None,
                    "stdout": "\n".join(output_lines),
                    "error": "No result returned from solve() function. "
                             "Stderr: " + (proc.stderr or "(empty)"),
                    "lp_content": self._read_lp_file(tmpdir),
                }

            return {
                "result": result,
                "stdout": "\n".join(output_lines),
                "error": None,
                "lp_content": self._read_lp_file(tmpdir),
            }

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        """Remove markdown code fences if the LLM wrapped its output in them."""
        lines = code.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    @staticmethod
    def _read_lp_file(tmpdir: str) -> str:
        """Read model.lp from the temp directory if it exists."""
        lp_path = Path(tmpdir) / "model.lp"
        if lp_path.is_file():
            try:
                return lp_path.read_text(encoding="utf-8")
            except Exception:
                return ""
        return ""

    def _build_wrapper(self, code: str, time_limit: int | None = None, show_solver_log: bool = False) -> str:
        """Build the wrapper script that imports and calls solve()."""
        kwargs = []
        if time_limit is not None:
            kwargs.append(f"time_limit={time_limit}")
        kwargs.append(f"show_solver_log={show_solver_log}")
        call = f"solve(data, {', '.join(kwargs)})"
        return textwrap.dedent(f"""\
            import json
            import sys

            # --- User-generated solver code ---
            {textwrap.indent(code, "            ").strip()}
            # --- End solver code ---

            if __name__ == "__main__":
                import traceback as _tb
                with open("data.json", encoding="utf-8") as _f:
                    data = json.load(_f)
                try:
                    result = {call}
                    print("__ORPILOT_RESULT__:" + json.dumps(result))
                except Exception as e:
                    print("__ORPILOT_RESULT__:" + json.dumps({{"status": "error", "error": _tb.format_exc()}}))
                    sys.exit(0)
        """)

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

    def execute(self, code: str, data: dict) -> dict:
        """Run the generated code's `solve(data)` function in a subprocess.

        Returns a dict with keys:
            - result: the dict returned by solve(data), or None
            - stdout: captured stdout
            - error: error message string, or None
        """
        wrapper = self._build_wrapper(code, data)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "solver_script.py"
            script_path.write_text(wrapper)

            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return {
                    "result": None,
                    "stdout": "",
                    "error": f"Execution timed out after {self.timeout}s",
                }

            if proc.returncode != 0:
                return {
                    "result": None,
                    "stdout": proc.stdout,
                    "error": proc.stderr or f"Process exited with code {proc.returncode}",
                }

            # Parse the JSON result from stdout
            stdout_lines = proc.stdout.strip().split("\n")
            # The last line should be our JSON marker
            result = None
            output_lines = []
            for line in stdout_lines:
                if line.startswith("__AIFOR_RESULT__:"):
                    try:
                        result = json.loads(line[len("__AIFOR_RESULT__:"):])
                    except json.JSONDecodeError:
                        pass
                else:
                    output_lines.append(line)

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
    def _read_lp_file(tmpdir: str) -> str:
        """Read model.lp from the temp directory if it exists."""
        lp_path = Path(tmpdir) / "model.lp"
        if lp_path.is_file():
            try:
                return lp_path.read_text(encoding="utf-8")
            except Exception:
                return ""
        return ""

    def _build_wrapper(self, code: str, data: dict) -> str:
        """Build the wrapper script that imports and calls solve()."""
        data_json = json.dumps(data)
        return textwrap.dedent(f"""\
            import json
            import sys

            # --- User-generated solver code ---
            {textwrap.indent(code, "            ").strip()}
            # --- End solver code ---

            if __name__ == "__main__":
                data = json.loads({data_json!r})
                try:
                    result = solve(data)
                    print("__AIFOR_RESULT__:" + json.dumps(result))
                except Exception as e:
                    print("__AIFOR_RESULT__:" + json.dumps({{"status": "error", "error": str(e)}}))
                    sys.exit(0)
        """)

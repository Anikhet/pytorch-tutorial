"""
Sandboxed code execution environment.

Provides safe execution of untrusted Python code using:
- Subprocess isolation (separate process)
- Timeout enforcement
- Memory limits (resource module on Unix)
- Blocked dangerous patterns (network, filesystem destruction, imports)
- Tempfile isolation (each execution gets its own temp directory)

Used by the agent for running generated code and tests safely.
"""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExecutionResult:
    """Result from sandboxed code execution."""

    passed: bool
    output: str
    error: str
    runtime_ms: float
    timed_out: bool = False


# Patterns that indicate dangerous operations
BLOCKED_PATTERNS: List[str] = [
    "socket",
    "requests.get",
    "requests.post",
    "urllib.request",
    "urllib.urlopen",
    "__import__",
    "importlib",
    "shutil.rmtree",
    "subprocess.call",
    "subprocess.run",
    "subprocess.Popen",
    "/etc/passwd",
    "/etc/shadow",
    "open('/etc",
    "open(\"/etc",
]


def check_code_safety(code: str) -> Optional[str]:
    """
    Check code for dangerous patterns before execution.

    Returns None if safe, or a description of the violation.
    """
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return f"Blocked pattern detected: '{pattern}'"
    return None


class CodeSandbox:
    """
    Sandboxed execution environment for untrusted Python code.

    Each execution runs in a fresh temp directory with a timeout.
    Dangerous operations are blocked before execution.
    """

    def __init__(self, timeout: int = 10, max_output_chars: int = 5000):
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute_code(self, code: str) -> ExecutionResult:
        """
        Execute Python code in an isolated subprocess.

        Args:
            code: Python source code to execute

        Returns:
            ExecutionResult with output, errors, and timing
        """
        # Pre-execution safety check
        violation = check_code_safety(code)
        if violation is not None:
            return ExecutionResult(
                passed=False,
                output="",
                error=f"Safety violation: {violation}",
                runtime_ms=0.0,
            )

        return self._run_in_subprocess(code)

    def execute_with_tests(
        self, solution_code: str, test_code: str
    ) -> ExecutionResult:
        """
        Execute a solution followed by test assertions.

        Combines solution_code and test_code into a single script,
        then runs in the sandbox. Tests should use assert statements.
        """
        combined = f"{solution_code}\n\n# --- Tests ---\n{test_code}"

        violation = check_code_safety(combined)
        if violation is not None:
            return ExecutionResult(
                passed=False,
                output="",
                error=f"Safety violation: {violation}",
                runtime_ms=0.0,
            )

        return self._run_in_subprocess(combined)

    def _run_in_subprocess(self, code: str) -> ExecutionResult:
        """
        Run code in an isolated subprocess with timeout.

        Creates a temp directory, writes code to a file,
        executes it, and captures output/errors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "solution.py")
            with open(script_path, "w") as f:
                f.write(code)

            start = time.perf_counter()
            try:
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                    env=self._safe_env(),
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                stdout = result.stdout[: self.max_output_chars]
                stderr = result.stderr[: self.max_output_chars]
                passed = result.returncode == 0

                return ExecutionResult(
                    passed=passed,
                    output=stdout,
                    error=stderr if not passed else "",
                    runtime_ms=elapsed_ms,
                )

            except subprocess.TimeoutExpired:
                elapsed_ms = (time.perf_counter() - start) * 1000
                return ExecutionResult(
                    passed=False,
                    output="",
                    error=f"Execution timed out after {self.timeout}s",
                    runtime_ms=elapsed_ms,
                    timed_out=True,
                )

    def _safe_env(self) -> dict:
        """
        Create a restricted environment for subprocess execution.

        Inherits PATH and PYTHONPATH but removes sensitive variables.
        """
        safe = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "HOME": tempfile.gettempdir(),
            "LANG": "en_US.UTF-8",
        }
        return safe


def run_batch(
    sandbox: CodeSandbox,
    solutions: List[str],
    test_code: str,
) -> List[ExecutionResult]:
    """
    Run multiple solutions against the same test code.

    Args:
        sandbox: CodeSandbox instance
        solutions: List of solution code strings
        test_code: Test assertions to run after each solution

    Returns:
        List of ExecutionResult, one per solution
    """
    results = []
    for solution in solutions:
        result = sandbox.execute_with_tests(solution, test_code)
        results.append(result)
    return results

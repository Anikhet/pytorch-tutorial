"""
Coding agent with ReAct and Plan-and-Solve modes.

The agent receives a coding task, reasons about it, selects tools,
executes them, observes results, and iterates until solved.

Two modes:
- ReAct: Think -> Act -> Observe loop (reactive, step-by-step)
- PlanAndSolve: Decompose -> Execute steps -> Integrate (structured)

Uses a TemplateBackend by default (no LLM required) with a clean
abstraction for swapping in a real LLM backend.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from tools import ToolRegistry, create_default_registry
from sandbox import CodeSandbox


class AgentMode(Enum):
    """Agent reasoning modes."""

    REACT = "react"
    PLAN_AND_SOLVE = "plan_and_solve"


@dataclass
class AgentStep:
    """A single step in the agent's execution trace."""

    step_num: int
    thought: str
    action: str
    action_input: Dict
    observation: str
    timestamp: float = 0.0


@dataclass
class AgentResult:
    """Final result from agent execution."""

    task: str
    solution: str
    steps: List[AgentStep] = field(default_factory=list)
    success: bool = False
    total_time_ms: float = 0.0
    n_attempts: int = 0


# --- Solution Templates ---
# Module-level dict populated after helper functions are defined.

_FIBONACCI = (
    "def fibonacci(n: int) -> int:\n"
    "    if n <= 1:\n"
    "        return n\n"
    "    a, b = 0, 1\n"
    "    for _ in range(2, n + 1):\n"
    "        a, b = b, a + b\n"
    "    return b\n"
)

_PALINDROME = (
    "def is_palindrome(s: str) -> bool:\n"
    "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
    "    return cleaned == cleaned[::-1]\n"
)

_FACTORIAL = (
    "def factorial(n: int) -> int:\n"
    "    if n < 0:\n"
    "        raise ValueError('n must be non-negative')\n"
    "    result = 1\n"
    "    for i in range(2, n + 1):\n"
    "        result *= i\n"
    "    return result\n"
)

_TWO_SUM = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, num in enumerate(nums):\n"
    "        complement = target - num\n"
    "        if complement in seen:\n"
    "            return [seen[complement], i]\n"
    "        seen[num] = i\n"
    "    return []\n"
)

_BINARY_SEARCH = (
    "def binary_search(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        elif arr[mid] < target:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid - 1\n"
    "    return -1\n"
)

_REVERSE = "def reverse_string(s: str) -> str:\n    return s[::-1]\n"

_SORT = (
    "def merge_sort(arr):\n"
    "    if len(arr) <= 1:\n"
    "        return arr\n"
    "    mid = len(arr) // 2\n"
    "    left = merge_sort(arr[:mid])\n"
    "    right = merge_sort(arr[mid:])\n"
    "    return _merge(left, right)\n\n"
    "def _merge(left, right):\n"
    "    result, i, j = [], 0, 0\n"
    "    while i < len(left) and j < len(right):\n"
    "        if left[i] <= right[j]:\n"
    "            result.append(left[i]); i += 1\n"
    "        else:\n"
    "            result.append(right[j]); j += 1\n"
    "    result.extend(left[i:])\n"
    "    result.extend(right[j:])\n"
    "    return result\n"
)

_PRIME = (
    "def is_prime(n: int) -> bool:\n"
    "    if n < 2:\n"
    "        return False\n"
    "    if n < 4:\n"
    "        return True\n"
    "    if n % 2 == 0 or n % 3 == 0:\n"
    "        return False\n"
    "    i = 5\n"
    "    while i * i <= n:\n"
    "        if n % i == 0 or n % (i + 2) == 0:\n"
    "            return False\n"
    "        i += 6\n"
    "    return True\n"
)

# Keyword -> template mapping
SOLUTION_TEMPLATES: Dict[str, str] = {
    "fibonacci": _FIBONACCI,
    "palindrome": _PALINDROME,
    "factorial": _FACTORIAL,
    "two_sum": _TWO_SUM,
    "binary_search": _BINARY_SEARCH,
    "reverse": _REVERSE,
    "sort": _SORT,
    "prime": _PRIME,
}


class GenerationBackend:
    """
    Abstract backend for code generation.

    Subclass this to plug in a real LLM (HuggingFace, API, etc.).
    The default TemplateBackend uses pattern matching for demos.
    """

    def generate(self, prompt: str, context: str = "") -> str:
        """Generate a response given a prompt and context."""
        raise NotImplementedError


class TemplateBackend(GenerationBackend):
    """
    Template-based code generation for self-contained demos.

    Uses keyword matching on the task description to select
    pre-built solution templates. In production, replace with
    an LLM backend.
    """

    def generate(self, prompt: str, context: str = "") -> str:
        """Match task keywords to return a template solution."""
        prompt_lower = prompt.lower()
        for keyword, template in SOLUTION_TEMPLATES.items():
            if keyword in prompt_lower:
                return template
        # Fallback: stub function
        return (
            "def solution(*args, **kwargs):\n"
            "    # TODO: Replace with LLM-generated solution\n"
            "    pass\n"
        )


class CodingAgent:
    """
    Main coding agent that solves programming tasks.

    Uses a generate-execute-observe loop with self-debugging.
    Supports ReAct and Plan-and-Solve modes.
    """

    def __init__(
        self,
        backend: GenerationBackend = None,
        registry: ToolRegistry = None,
        sandbox: CodeSandbox = None,
        mode: AgentMode = AgentMode.REACT,
        max_attempts: int = 3,
    ):
        self.backend = backend or TemplateBackend()
        self.registry = registry or create_default_registry()
        self.sandbox = sandbox or CodeSandbox()
        self.mode = mode
        self.max_attempts = max_attempts

    def solve(self, task: str, test_code: str = "") -> AgentResult:
        """
        Solve a coding task with self-debugging.

        Args:
            task: Natural language task description
            test_code: Optional test assertions to validate solution

        Returns:
            AgentResult with solution, steps, and success status
        """
        if self.mode == AgentMode.PLAN_AND_SOLVE:
            return self._solve_plan_and_solve(task, test_code)
        return self._solve_react(task, test_code)

    def _solve_react(self, task: str, test_code: str) -> AgentResult:
        """ReAct-style: Think -> Generate -> Test -> Fix loop."""
        start = time.perf_counter()
        steps: List[AgentStep] = []
        solution = ""

        for attempt in range(1, self.max_attempts + 1):
            # Think: generate or fix
            context = self._build_context(task, steps)
            if steps and not steps[-1].observation.startswith("PASS"):
                thought = f"Attempt {attempt}: Fixing based on error"
            else:
                thought = f"Attempt {attempt}: Generating solution"

            solution = self.backend.generate(task, context)

            # Act: execute in sandbox
            if test_code:
                exec_result = self.sandbox.execute_with_tests(
                    solution, test_code
                )
            else:
                exec_result = self.sandbox.execute_code(solution)

            # Observe: record result
            observation = (
                "PASSED" if exec_result.passed
                else f"FAILED: {exec_result.error}"
            )

            steps.append(AgentStep(
                step_num=attempt,
                thought=thought,
                action="execute_code",
                action_input={"code": solution[:200] + "..."},
                observation=observation,
                timestamp=time.perf_counter() - start,
            ))

            if exec_result.passed:
                elapsed = (time.perf_counter() - start) * 1000
                return AgentResult(
                    task=task, solution=solution, steps=steps,
                    success=True, total_time_ms=elapsed,
                    n_attempts=attempt,
                )

        elapsed = (time.perf_counter() - start) * 1000
        return AgentResult(
            task=task, solution=solution, steps=steps,
            success=False, total_time_ms=elapsed,
            n_attempts=self.max_attempts,
        )

    def _solve_plan_and_solve(
        self, task: str, test_code: str
    ) -> AgentResult:
        """Plan-and-Solve: Decompose task, solve each part."""
        start = time.perf_counter()
        steps: List[AgentStep] = []

        # Step 1: Plan
        steps.append(AgentStep(
            step_num=1,
            thought="Decomposing task into sub-problems",
            action="plan",
            action_input={"task": task},
            observation="Plan: 1) Parse requirements 2) Generate 3) Test",
            timestamp=time.perf_counter() - start,
        ))

        # Step 2: Generate + Test
        solution = self.backend.generate(task)
        if test_code:
            exec_result = self.sandbox.execute_with_tests(solution, test_code)
        else:
            exec_result = self.sandbox.execute_code(solution)

        observation = (
            "PASSED" if exec_result.passed
            else f"FAILED: {exec_result.error}"
        )

        steps.append(AgentStep(
            step_num=2,
            thought="Executing plan",
            action="execute_code",
            action_input={"code": solution[:200] + "..."},
            observation=observation,
            timestamp=time.perf_counter() - start,
        ))

        elapsed = (time.perf_counter() - start) * 1000
        return AgentResult(
            task=task, solution=solution, steps=steps,
            success=exec_result.passed, total_time_ms=elapsed,
            n_attempts=1,
        )

    def _build_context(self, task: str, steps: List[AgentStep]) -> str:
        """Build context string from previous steps for self-debugging."""
        if not steps:
            return f"Task: {task}"
        last = steps[-1]
        return (
            f"Task: {task}\n"
            f"Previous attempt failed with: {last.observation}\n"
            f"Fix the error and try again."
        )

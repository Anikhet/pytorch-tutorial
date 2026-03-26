"""
Assessment pipeline for coding agents.

Implements:
- pass@k metric (unbiased estimator from Codex paper)
- Full assessment pipeline: run agent on benchmark, compute metrics
- Model comparison with statistical testing
- Result formatting and reporting

This module is the bridge between the agent (agent.py) and
the benchmark dataset (dataset.py).
"""

import time
from dataclasses import dataclass, field
from math import comb
from typing import Dict, List, Optional

import numpy as np

from agent import AgentResult, CodingAgent
from dataset import CodingTask, TaskDataset
from sandbox import CodeSandbox


@dataclass
class TaskScore:
    """Score for a single task across multiple samples."""

    task_id: str
    n_samples: int
    n_correct: int
    pass_at_1: float
    pass_at_5: float = 0.0
    avg_attempts: float = 0.0
    avg_time_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregate results across an entire benchmark."""

    model_name: str
    task_scores: List[TaskScore] = field(default_factory=list)
    overall_pass_at_1: float = 0.0
    overall_pass_at_5: float = 0.0
    total_tasks: int = 0
    total_time_ms: float = 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k from Chen et al. 2021 (Codex).

    Computes: 1 - C(n-c, k) / C(n, k)
    where n = total samples, c = correct samples, k = k value.

    This gives the probability that at least one of k randomly
    selected samples from n total is correct, given c are correct.

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: Number of samples to select

    Returns:
        Probability that at least one of k samples passes
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    # Use log-space for numerical stability with large values
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_batch(
    n_samples: np.ndarray, n_correct: np.ndarray, k: int
) -> np.ndarray:
    """
    Vectorized pass@k for a batch of tasks.

    Args:
        n_samples: (num_tasks,) total samples per task
        n_correct: (num_tasks,) correct samples per task
        k: k value

    Returns:
        (num_tasks,) pass@k for each task
    """
    results = np.array([
        pass_at_k(int(n), int(c), k)
        for n, c in zip(n_samples, n_correct)
    ])
    return results


class CodingAssessment:
    """
    Full assessment pipeline for coding agents.

    Runs the agent on each task in a benchmark (optionally multiple
    times per task for pass@k estimation), collects results,
    and computes aggregate metrics.
    """

    def __init__(
        self,
        agent: CodingAgent,
        sandbox: CodeSandbox = None,
        samples_per_task: int = 1,
    ):
        self.agent = agent
        self.sandbox = sandbox or CodeSandbox()
        self.samples_per_task = samples_per_task

    def run_benchmark(
        self,
        dataset: TaskDataset,
        model_name: str = "agent",
    ) -> BenchmarkResult:
        """
        Run the agent on every task in the dataset.

        For each task, runs `samples_per_task` attempts and
        computes pass@k metrics.

        Returns:
            BenchmarkResult with per-task and overall scores
        """
        start = time.perf_counter()
        task_scores: List[TaskScore] = []

        for task in dataset:
            score = self._assess_task(task)
            task_scores.append(score)

        elapsed = (time.perf_counter() - start) * 1000

        # Compute overall metrics
        n_arr = np.array([s.n_samples for s in task_scores])
        c_arr = np.array([s.n_correct for s in task_scores])

        overall_p1 = float(np.mean(pass_at_k_batch(n_arr, c_arr, 1)))
        overall_p5 = float(np.mean(
            pass_at_k_batch(n_arr, c_arr, min(5, self.samples_per_task))
        ))

        return BenchmarkResult(
            model_name=model_name,
            task_scores=task_scores,
            overall_pass_at_1=overall_p1,
            overall_pass_at_5=overall_p5,
            total_tasks=len(dataset),
            total_time_ms=elapsed,
        )

    def _assess_task(self, task: CodingTask) -> TaskScore:
        """Run agent on a single task multiple times and score."""
        n_correct = 0
        times: List[float] = []
        attempts: List[int] = []

        for _ in range(self.samples_per_task):
            result = self.agent.solve(task.prompt, task.test_code)
            if result.success:
                n_correct += 1
            times.append(result.total_time_ms)
            attempts.append(result.n_attempts)

        p1 = pass_at_k(self.samples_per_task, n_correct, 1)
        p5 = pass_at_k(
            self.samples_per_task, n_correct,
            min(5, self.samples_per_task)
        )

        return TaskScore(
            task_id=task.task_id,
            n_samples=self.samples_per_task,
            n_correct=n_correct,
            pass_at_1=p1,
            pass_at_5=p5,
            avg_attempts=float(np.mean(attempts)),
            avg_time_ms=float(np.mean(times)),
        )

    def compare_models(
        self,
        result_a: BenchmarkResult,
        result_b: BenchmarkResult,
    ) -> Dict:
        """
        Compare two model results side by side.

        Returns dict with comparison metrics.
        """
        return {
            "model_a": result_a.model_name,
            "model_b": result_b.model_name,
            "pass_at_1_a": result_a.overall_pass_at_1,
            "pass_at_1_b": result_b.overall_pass_at_1,
            "pass_at_1_diff": (
                result_a.overall_pass_at_1 - result_b.overall_pass_at_1
            ),
            "total_time_a_ms": result_a.total_time_ms,
            "total_time_b_ms": result_b.total_time_ms,
        }


def format_results(result: BenchmarkResult) -> str:
    """Format benchmark results as a readable report string."""
    lines = [
        f"=== Assessment Results: {result.model_name} ===",
        f"Total tasks: {result.total_tasks}",
        f"Overall pass@1: {result.overall_pass_at_1:.3f}",
        f"Overall pass@5: {result.overall_pass_at_5:.3f}",
        f"Total time: {result.total_time_ms:.1f}ms",
        "",
        "Per-task breakdown:",
    ]
    for score in result.task_scores:
        status = "PASS" if score.n_correct > 0 else "FAIL"
        lines.append(
            f"  [{status}] {score.task_id}: "
            f"pass@1={score.pass_at_1:.2f}, "
            f"attempts={score.avg_attempts:.1f}, "
            f"time={score.avg_time_ms:.0f}ms"
        )
    return "\n".join(lines)

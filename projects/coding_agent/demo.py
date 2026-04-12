"""
Interactive demo for the coding agent.

Provides three interfaces:
1. CLI mode: Run agent on a single task from command line
2. Benchmark mode: Run full benchmark and print results
3. Gradio mode: Interactive web UI (requires gradio)

Usage:
    python demo.py --mode cli --task "Write fibonacci function"
    python demo.py --mode benchmark
    python demo.py --mode gradio
"""

import argparse
import sys
from typing import List

from agent import CodingAgent, AgentMode, AgentResult
from dataset import TaskDataset
from evaluator import CodingAssessment, format_results
from sandbox import CodeSandbox


def print_agent_trace(result: AgentResult) -> None:
    """Print a step-by-step trace of agent execution."""
    print(f"\n{'='*60}")
    print(f"Task: {result.task}")
    print(f"{'='*60}")

    for step in result.steps:
        print(f"\n--- Step {step.step_num} ---")
        print(f"  Thought: {step.thought}")
        print(f"  Action:  {step.action}")
        print(f"  Result:  {step.observation}")
        print(f"  Time:    {step.timestamp:.3f}s")

    print(f"\n{'='*60}")
    status = "SUCCESS" if result.success else "FAILED"
    print(f"Status: {status}")
    print(f"Attempts: {result.n_attempts}")
    print(f"Total time: {result.total_time_ms:.1f}ms")

    if result.success:
        print(f"\nSolution:\n{result.solution}")


def run_cli_mode(task: str, test_code: str = "") -> None:
    """Run agent on a single task and print trace."""
    agent = CodingAgent(max_attempts=3)
    result = agent.solve(task, test_code)
    print_agent_trace(result)


def run_benchmark_mode(benchmark: str = "mini") -> None:
    """Run agent on benchmark and print results."""
    dataset = (
        TaskDataset.mini() if benchmark == "mini"
        else TaskDataset.medium()
    )

    print(f"Running {benchmark} benchmark ({len(dataset)} tasks)...")
    agent = CodingAgent(max_attempts=3)
    assessment = CodingAssessment(agent, samples_per_task=1)
    result = assessment.run_benchmark(dataset, model_name="TemplateAgent")
    print(format_results(result))


def run_comparison_mode() -> None:
    """Compare ReAct vs Plan-and-Solve on the mini benchmark."""
    dataset = TaskDataset.mini()

    print("Comparing ReAct vs Plan-and-Solve...")
    print()

    # ReAct agent
    react_agent = CodingAgent(
        mode=AgentMode.REACT, max_attempts=3
    )
    react_assessment = CodingAssessment(react_agent, samples_per_task=1)
    react_result = react_assessment.run_benchmark(
        dataset, model_name="ReAct"
    )

    # Plan-and-Solve agent
    plan_agent = CodingAgent(
        mode=AgentMode.PLAN_AND_SOLVE, max_attempts=1
    )
    plan_assessment = CodingAssessment(plan_agent, samples_per_task=1)
    plan_result = plan_assessment.run_benchmark(
        dataset, model_name="PlanAndSolve"
    )

    print(format_results(react_result))
    print()
    print(format_results(plan_result))

    comparison = react_assessment.compare_models(react_result, plan_result)
    print(f"\nComparison:")
    print(f"  ReAct pass@1:        {comparison['pass_at_1_a']:.3f}")
    print(f"  PlanAndSolve pass@1: {comparison['pass_at_1_b']:.3f}")
    print(f"  Difference:          {comparison['pass_at_1_diff']:+.3f}")


def create_gradio_app():
    """Create Gradio interface for interactive agent demo."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        sys.exit(1)

    agent = CodingAgent(max_attempts=3)

    def solve_task(task_text: str, test_code: str) -> str:
        """Solve a task and return formatted results."""
        if not task_text.strip():
            return "Please enter a task description."

        result = agent.solve(task_text, test_code)
        lines = []
        lines.append(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        lines.append(f"Attempts: {result.n_attempts}")
        lines.append(f"Time: {result.total_time_ms:.1f}ms")
        lines.append("")

        for step in result.steps:
            lines.append(f"--- Step {step.step_num} ---")
            lines.append(f"Thought: {step.thought}")
            lines.append(f"Result: {step.observation}")
            lines.append("")

        if result.success:
            lines.append("Solution:")
            lines.append(result.solution)
        return "\n".join(lines)

    def run_benchmark_tab(benchmark_name: str) -> str:
        """Run benchmark and return results."""
        dataset = (
            TaskDataset.mini() if benchmark_name == "Mini (5 tasks)"
            else TaskDataset.medium()
        )
        assessment = CodingAssessment(agent, samples_per_task=1)
        result = assessment.run_benchmark(dataset, "TemplateAgent")
        return format_results(result)

    with gr.Blocks(title="Coding Agent Demo") as app:
        gr.Markdown("# Coding Agent Demo")
        gr.Markdown(
            "Interactive coding agent with self-debugging "
            "and sandboxed execution."
        )

        with gr.Tab("Solve Task"):
            task_input = gr.Textbox(
                label="Task Description",
                placeholder="Write a function fibonacci(n)...",
                lines=3,
            )
            test_input = gr.Textbox(
                label="Test Code (optional)",
                placeholder="assert fibonacci(10) == 55",
                lines=3,
            )
            solve_btn = gr.Button("Solve", variant="primary")
            output = gr.Textbox(label="Results", lines=20)
            solve_btn.click(solve_task, [task_input, test_input], output)

        with gr.Tab("Benchmark"):
            bench_choice = gr.Radio(
                ["Mini (5 tasks)", "Medium (10 tasks)"],
                label="Benchmark",
                value="Mini (5 tasks)",
            )
            bench_btn = gr.Button("Run Benchmark", variant="primary")
            bench_output = gr.Textbox(label="Results", lines=20)
            bench_btn.click(run_benchmark_tab, bench_choice, bench_output)

    return app


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Coding Agent Demo")
    parser.add_argument(
        "--mode",
        choices=["cli", "benchmark", "compare", "gradio"],
        default="benchmark",
    )
    parser.add_argument("--task", type=str, default="Write fibonacci(n)")
    parser.add_argument("--test", type=str, default="")
    parser.add_argument(
        "--benchmark", choices=["mini", "medium"], default="mini"
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli_mode(args.task, args.test)
    elif args.mode == "benchmark":
        run_benchmark_mode(args.benchmark)
    elif args.mode == "compare":
        run_comparison_mode()
    elif args.mode == "gradio":
        app = create_gradio_app()
        app.launch()


if __name__ == "__main__":
    main()

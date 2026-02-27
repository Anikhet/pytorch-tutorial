"""
Task dataset for coding agent training and assessment.

Provides:
- CodingTask dataclass for individual problems
- TaskDataset with built-in mini and medium benchmarks
- JSONL loading for custom benchmarks
- Train/test splitting

Each task has a prompt, entry_point, canonical_solution,
and test assertions for pass/fail verification.
"""

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CodingTask:
    """A single coding task with tests."""

    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test_code: str
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# --- Built-in Benchmark Problems ---

def _create_mini_benchmark() -> List[CodingTask]:
    """5 easy problems for quick testing."""
    return [
        CodingTask(
            task_id="mini_001",
            prompt="Write a function fibonacci(n) that returns the nth fibonacci number.",
            entry_point="fibonacci",
            canonical_solution=(
                "def fibonacci(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for _ in range(2, n + 1):\n"
                "        a, b = b, a + b\n"
                "    return b\n"
            ),
            test_code=(
                "assert fibonacci(0) == 0\n"
                "assert fibonacci(1) == 1\n"
                "assert fibonacci(10) == 55\n"
            ),
            difficulty="easy",
            tags=["math", "dp"],
        ),
        CodingTask(
            task_id="mini_002",
            prompt="Write a function is_palindrome(s) that checks if a string is a palindrome.",
            entry_point="is_palindrome",
            canonical_solution=(
                "def is_palindrome(s):\n"
                "    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n"
                "    return cleaned == cleaned[::-1]\n"
            ),
            test_code=(
                "assert is_palindrome('racecar') == True\n"
                "assert is_palindrome('hello') == False\n"
                "assert is_palindrome('A man a plan a canal Panama') == True\n"
            ),
            difficulty="easy",
            tags=["string"],
        ),
        CodingTask(
            task_id="mini_003",
            prompt="Write a function two_sum(nums, target) that returns indices of two numbers that add up to target.",
            entry_point="two_sum",
            canonical_solution=(
                "def two_sum(nums, target):\n"
                "    seen = {}\n"
                "    for i, num in enumerate(nums):\n"
                "        complement = target - num\n"
                "        if complement in seen:\n"
                "            return [seen[complement], i]\n"
                "        seen[num] = i\n"
                "    return []\n"
            ),
            test_code=(
                "assert two_sum([2, 7, 11, 15], 9) == [0, 1]\n"
                "assert two_sum([3, 2, 4], 6) == [1, 2]\n"
            ),
            difficulty="easy",
            tags=["array", "hashmap"],
        ),
        CodingTask(
            task_id="mini_004",
            prompt="Write a function factorial(n) that returns n factorial.",
            entry_point="factorial",
            canonical_solution=(
                "def factorial(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('n must be non-negative')\n"
                "    result = 1\n"
                "    for i in range(2, n + 1):\n"
                "        result *= i\n"
                "    return result\n"
            ),
            test_code=(
                "assert factorial(0) == 1\n"
                "assert factorial(5) == 120\n"
                "assert factorial(10) == 3628800\n"
            ),
            difficulty="easy",
            tags=["math"],
        ),
        CodingTask(
            task_id="mini_005",
            prompt="Write a function binary_search(arr, target) that returns the index of target in sorted array, or -1.",
            entry_point="binary_search",
            canonical_solution=(
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
            ),
            test_code=(
                "assert binary_search([1, 3, 5, 7, 9], 5) == 2\n"
                "assert binary_search([1, 3, 5, 7, 9], 4) == -1\n"
                "assert binary_search([], 1) == -1\n"
            ),
            difficulty="easy",
            tags=["search"],
        ),
    ]


def _create_medium_benchmark() -> List[CodingTask]:
    """5 medium problems for fuller assessment."""
    return [
        CodingTask(
            task_id="med_001",
            prompt="Write a function longest_common_subsequence(s1, s2) returning the length of the LCS.",
            entry_point="longest_common_subsequence",
            canonical_solution=(
                "def longest_common_subsequence(s1, s2):\n"
                "    m, n = len(s1), len(s2)\n"
                "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
                "    for i in range(1, m + 1):\n"
                "        for j in range(1, n + 1):\n"
                "            if s1[i-1] == s2[j-1]:\n"
                "                dp[i][j] = dp[i-1][j-1] + 1\n"
                "            else:\n"
                "                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
                "    return dp[m][n]\n"
            ),
            test_code=(
                "assert longest_common_subsequence('abcde', 'ace') == 3\n"
                "assert longest_common_subsequence('abc', 'def') == 0\n"
                "assert longest_common_subsequence('', 'abc') == 0\n"
            ),
            difficulty="medium",
            tags=["dp", "string"],
        ),
        CodingTask(
            task_id="med_002",
            prompt="Write a function valid_parentheses(s) that checks if brackets ()[]{}  are balanced.",
            entry_point="valid_parentheses",
            canonical_solution=(
                "def valid_parentheses(s):\n"
                "    stack = []\n"
                "    mapping = {')': '(', ']': '[', '}': '{'}\n"
                "    for char in s:\n"
                "        if char in mapping.values():\n"
                "            stack.append(char)\n"
                "        elif char in mapping:\n"
                "            if not stack or stack[-1] != mapping[char]:\n"
                "                return False\n"
                "            stack.pop()\n"
                "    return len(stack) == 0\n"
            ),
            test_code=(
                "assert valid_parentheses('()[]{}') == True\n"
                "assert valid_parentheses('(]') == False\n"
                "assert valid_parentheses('{[]}') == True\n"
                "assert valid_parentheses('') == True\n"
            ),
            difficulty="medium",
            tags=["stack"],
        ),
        CodingTask(
            task_id="med_003",
            prompt="Write a function merge_intervals(intervals) that merges overlapping intervals.",
            entry_point="merge_intervals",
            canonical_solution=(
                "def merge_intervals(intervals):\n"
                "    if not intervals:\n"
                "        return []\n"
                "    intervals.sort(key=lambda x: x[0])\n"
                "    merged = [intervals[0]]\n"
                "    for start, end in intervals[1:]:\n"
                "        if start <= merged[-1][1]:\n"
                "            merged[-1] = [merged[-1][0], max(merged[-1][1], end)]\n"
                "        else:\n"
                "            merged.append([start, end])\n"
                "    return merged\n"
            ),
            test_code=(
                "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
                "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
                "assert merge_intervals([]) == []\n"
            ),
            difficulty="medium",
            tags=["array", "sorting"],
        ),
        CodingTask(
            task_id="med_004",
            prompt="Write a function max_subarray_sum(nums) using Kadane's algorithm.",
            entry_point="max_subarray_sum",
            canonical_solution=(
                "def max_subarray_sum(nums):\n"
                "    if not nums:\n"
                "        return 0\n"
                "    max_sum = current = nums[0]\n"
                "    for num in nums[1:]:\n"
                "        current = max(num, current + num)\n"
                "        max_sum = max(max_sum, current)\n"
                "    return max_sum\n"
            ),
            test_code=(
                "assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6\n"
                "assert max_subarray_sum([1]) == 1\n"
                "assert max_subarray_sum([-1, -2, -3]) == -1\n"
            ),
            difficulty="medium",
            tags=["array", "dp"],
        ),
        CodingTask(
            task_id="med_005",
            prompt="Write a function flatten_nested_list(lst) that flattens arbitrarily nested lists.",
            entry_point="flatten_nested_list",
            canonical_solution=(
                "def flatten_nested_list(lst):\n"
                "    result = []\n"
                "    for item in lst:\n"
                "        if isinstance(item, list):\n"
                "            result.extend(flatten_nested_list(item))\n"
                "        else:\n"
                "            result.append(item)\n"
                "    return result\n"
            ),
            test_code=(
                "assert flatten_nested_list([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]\n"
                "assert flatten_nested_list([]) == []\n"
                "assert flatten_nested_list([[1], [[2]], [[[3]]]]) == [1, 2, 3]\n"
            ),
            difficulty="medium",
            tags=["recursion"],
        ),
    ]


class TaskDataset:
    """
    Dataset of coding tasks for training and assessment.

    Includes built-in mini (5 easy) and medium (5 medium) benchmarks.
    Also supports loading custom tasks from JSONL files.
    """

    def __init__(self, tasks: List[CodingTask] = None):
        self.tasks = tasks or []

    @classmethod
    def mini(cls) -> "TaskDataset":
        """Load the built-in 5-problem mini benchmark."""
        return cls(_create_mini_benchmark())

    @classmethod
    def medium(cls) -> "TaskDataset":
        """Load the built-in 10-problem combined benchmark."""
        tasks = _create_mini_benchmark() + _create_medium_benchmark()
        return cls(tasks)

    @classmethod
    def from_jsonl(cls, path: str) -> "TaskDataset":
        """Load tasks from a JSONL file."""
        tasks = []
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                tasks.append(CodingTask(**data))
        return cls(tasks)

    def split(
        self, train_ratio: float = 0.8, seed: int = 42
    ) -> Tuple["TaskDataset", "TaskDataset"]:
        """Split into train/test datasets."""
        rng = random.Random(seed)
        shuffled = list(self.tasks)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        return (
            TaskDataset(shuffled[:split_idx]),
            TaskDataset(shuffled[split_idx:]),
        )

    def filter_by_difficulty(self, difficulty: str) -> "TaskDataset":
        """Return a new dataset with only tasks of given difficulty."""
        filtered = [t for t in self.tasks if t.difficulty == difficulty]
        return TaskDataset(filtered)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> CodingTask:
        return self.tasks[idx]

    def __iter__(self):
        return iter(self.tasks)

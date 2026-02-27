"""
Tool framework for the coding agent.

Provides a registry of tools that the agent can invoke:
- FileReadTool: Read file contents
- FileWriteTool: Write/create files
- TerminalTool: Run shell commands (sandboxed)
- CodeSearchTool: Regex search across files
- TestRunnerTool: Run pytest on a file

Each tool follows the BaseTool interface with name, description,
and execute(). The ToolRegistry manages available tools and
dispatches calls by name.
"""

import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ToolResult:
    """Result from a tool execution."""

    success: bool
    output: str
    error: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the agent."""

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""


class FileReadTool(BaseTool):
    """Read the contents of a file."""

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read a file. Args: path (str)"

    def execute(self, path: str = "", **kwargs) -> ToolResult:
        """Read file at the given path."""
        try:
            if not os.path.exists(path):
                return ToolResult(False, "", f"File not found: {path}")
            with open(path, "r") as f:
                content = f.read()
            return ToolResult(True, content)
        except Exception as e:
            return ToolResult(False, "", str(e))


class FileWriteTool(BaseTool):
    """Write content to a file."""

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Write to a file. Args: path (str), content (str)"

    def execute(
        self, path: str = "", content: str = "", **kwargs
    ) -> ToolResult:
        """Write content to the given path, creating dirs if needed."""
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return ToolResult(True, f"Written {len(content)} chars to {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))


class TerminalTool(BaseTool):
    """Run a shell command with timeout and safety checks."""

    BLOCKED_PATTERNS = [
        "rm -rf /",
        "mkfs",
        "dd if=",
        ":(){",
        "fork bomb",
        "shutdown",
        "reboot",
    ]

    @property
    def name(self) -> str:
        return "terminal"

    @property
    def description(self) -> str:
        return "Run a shell command. Args: command (str)"

    def execute(
        self, command: str = "", timeout: int = 30, **kwargs
    ) -> ToolResult:
        """Run command in subprocess with timeout and safety."""
        # Safety check
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in command:
                return ToolResult(
                    False, "", f"Blocked dangerous pattern: {pattern}"
                )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir(),
            )
            output = result.stdout[:5000]  # truncate long output
            if result.returncode != 0:
                return ToolResult(False, output, result.stderr[:2000])
            return ToolResult(True, output)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(False, "", str(e))


class CodeSearchTool(BaseTool):
    """Search for a regex pattern across files in a directory."""

    @property
    def name(self) -> str:
        return "code_search"

    @property
    def description(self) -> str:
        return "Regex search in files. Args: pattern (str), directory (str)"

    def execute(
        self, pattern: str = "", directory: str = ".", **kwargs
    ) -> ToolResult:
        """Search for pattern in all .py files under directory."""
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return ToolResult(False, "", f"Invalid regex: {e}")

        matches: List[str] = []
        try:
            for root, _dirs, files in os.walk(directory):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r") as f:
                            for i, line in enumerate(f, 1):
                                if compiled.search(line):
                                    matches.append(
                                        f"{fpath}:{i}: {line.rstrip()}"
                                    )
                    except (UnicodeDecodeError, PermissionError):
                        continue

            if not matches:
                return ToolResult(True, "No matches found.")
            return ToolResult(True, "\n".join(matches[:50]))
        except Exception as e:
            return ToolResult(False, "", str(e))


class TestRunnerTool(BaseTool):
    """Run Python tests on a file using subprocess."""

    @property
    def name(self) -> str:
        return "test_runner"

    @property
    def description(self) -> str:
        return "Run tests for a Python file. Args: test_file (str)"

    def execute(
        self, test_file: str = "", timeout: int = 60, **kwargs
    ) -> ToolResult:
        """Run pytest on the given test file."""
        if not os.path.exists(test_file):
            return ToolResult(False, "", f"Test file not found: {test_file}")

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout[:5000]
            if result.returncode != 0:
                return ToolResult(False, output, result.stderr[:2000])
            return ToolResult(True, output)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Tests timed out after {timeout}s")
        except Exception as e:
            return ToolResult(False, "", str(e))


class ToolRegistry:
    """Registry that manages available tools and dispatches calls."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        """Return list of {name, description} for all registered tools."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Dispatch execution to the named tool."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(False, "", f"Unknown tool: {tool_name}")
        return tool.execute(**kwargs)


def create_default_registry() -> ToolRegistry:
    """Create a registry with all default tools pre-registered."""
    registry = ToolRegistry()
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(TerminalTool())
    registry.register(CodeSearchTool())
    registry.register(TestRunnerTool())
    return registry

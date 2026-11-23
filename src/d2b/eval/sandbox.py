"""Run a CadQuery program written by a model, with guards, and return the solid it builds."""

# A model's program is untrusted. This is a set of guards, not a full security sandbox.

from __future__ import annotations

import ast
import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cadquery as cq

# A program may import these modules and nothing else.
ALLOWED_IMPORTS = {"cadquery", "math"}

# Built-in functions that can read files, run other code or reach outside the program.
FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}

# Parts of CadQuery that read or write files, or open a window.
FORBIDDEN_ATTRIBUTES = {"exporters", "importers", "vis", "occ_impl", "exportStep", "exportBrep"}

# Seconds a program may run before it is stopped.
DEFAULT_TIMEOUT = 30.0

# Largest file a program may write, in bytes.
MAX_FILE_BYTES = 50 * 1024 * 1024

# The script run in the separate process: it runs the program and saves the solid it built.
RUNNER = """
import sys
import cadquery as cq

namespace = {}
exec(compile(open("program.py").read(), "program.py", "exec"), namespace)
result = namespace.get("result")
if isinstance(result, cq.Workplane):
    solids = [s for v in result.vals() if isinstance(v, cq.Shape) for s in v.Solids()]
elif isinstance(result, cq.Shape):
    solids = result.Solids()
else:
    sys.exit(3)
if not solids:
    sys.exit(3)
shape = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
shape.exportBrep("result.brep")
"""


# What happened when a program was run, and the solid it built if it succeeded.
@dataclass(frozen=True)
class SandboxResult:
    status: str  # "ok", "rejected", "error", "timeout" or "no_result"
    detail: str  # the reason, when the status is not "ok"
    shape: cq.Shape | None  # the solid the program built, when the status is "ok"


# Reads a program without running it; returns why it is not allowed, or None if it is fine.
def check_program(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return f"syntax error: {error.msg} (line {error.lineno})"
    for node in ast.walk(tree):
        # Only CadQuery and the maths module may be imported.
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{alias.name}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute):
            # Names like __class__ are a common way to escape restrictions.
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                return f"use of '.{node.attr}' is not allowed"
    return None


# Applies limits to the separate process just before the program starts in it.
def _limit_process(timeout: float):
    # Runs inside the new process, before the program does.
    def apply() -> None:
        # Stop the process if it uses too much processor time or writes a huge file.
        cpu_seconds = int(timeout) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE_BYTES, MAX_FILE_BYTES))

    return apply


# Runs an untrusted program in a separate process and returns the solid it built.
def run_untrusted(code: str, timeout: float = DEFAULT_TIMEOUT) -> SandboxResult:
    # Guard 1: refuse programs that import other modules or touch files.
    problem = check_program(code)
    if problem is not None:
        return SandboxResult("rejected", problem, None)

    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        (work / "program.py").write_text(code)
        (work / "runner.py").write_text(RUNNER)
        try:
            # Guard 2: a separate process, in an empty folder, with a time limit.
            # Guard 3: an empty environment, so API keys and other secrets are not passed on.
            done = subprocess.run(  # noqa: S603
                [sys.executable, "-I", "runner.py"],
                cwd=work,
                env={"HOME": str(work), "PATH": os.defpath},
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limit_process(timeout),  # noqa: PLW1509
            )
        except subprocess.TimeoutExpired:
            return SandboxResult("timeout", f"stopped after {timeout:g} seconds", None)

        if done.returncode == 3:
            return SandboxResult("no_result", "the program did not define a solid `result`", None)
        if done.returncode != 0:
            # The last line of the error output is the exception that stopped the program.
            lines = done.stderr.strip().splitlines()
            return SandboxResult("error", lines[-1] if lines else "unknown error", None)
        # Load the solid while the folder still exists.
        return SandboxResult("ok", "", cq.Shape.importBrep(str(work / "result.brep")))

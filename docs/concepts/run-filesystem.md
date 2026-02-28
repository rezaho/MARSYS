# Run Filesystem (RunFileSystem)

MARSYS uses a **run-scoped virtual filesystem** to keep file paths consistent across tools, agents, and handoffs. This avoids the “browser downloads to one folder, file tools read from another” problem.

## Overview

`RunFileSystem` provides a **virtual POSIX path space** for each run:

- All agent-facing paths are **virtual** and **POSIX-style**.
- `/` is always the **run root** (the working root for that run).
- Paths are resolved with traversal protection (no escaping the run root).
- Optional **mounts** let you expose extra host folders at virtual prefixes.

This lets every agent speak the **same path language**, even if they operate in different host directories.

## Virtual Paths

Use **virtual paths** everywhere between tools and agents:

- Preferred (tool-returned): `./downloads/report.pdf`
- Also accepted: `/downloads/report.pdf`
- Relative paths like `./data/summary.txt` are resolved against `RunFileSystem.cwd`

Tools that write files return virtual paths so other agents can read them without guessing host paths.

## Creating a Run Filesystem

```python
from pathlib import Path
from marsys.environment.filesystem import RunFileSystem

run_root = Path("./runs/run-20260206")
fs = RunFileSystem.local(
    run_root=run_root,
    cwd="/",
    extra_mounts={
        "/datasets": Path("/shared/datasets"),
    },
)
```

Key options:

- `run_root`: host directory mounted at `/`
- `cwd`: initial virtual working directory
- `extra_mounts`: map additional host paths to virtual prefixes
- `memory_root`: convenience mount for `/memory`
- `allow_symlink_escape`: allow symlinks outside roots (off by default)

## Sharing Across Agents

To share files across agents, **use the same run root or the same RunFileSystem**:

```python
from marsys.agents import FileOperationAgent, CodeExecutionAgent, DataAnalysisAgent
from marsys.models import ModelConfig

fs = RunFileSystem.local(run_root=Path("./runs/run-20260206"))

file_agent = FileOperationAgent(model_config=config, name="files", filesystem=fs)
code_agent = CodeExecutionAgent(model_config=config, name="code", filesystem=fs)
data_agent = DataAnalysisAgent(model_config=config, name="data", filesystem=fs)
```

Browser automation also uses a run filesystem internally. If you set `tmp_dir` to the same run root, downloads and screenshots will land under the same virtual paths:

```python
browser = await BrowserAgent.create_safe(
    model_config=config,
    name="browser",
    tmp_dir="./runs/run-20260206"
)
# Downloads will appear under ./downloads, screenshots under ./screenshots
```

If you need custom mounts to be visible to the browser too, pass the same `RunFileSystem` to `BrowserTool` or `BrowserAgent` (advanced usage).

## Default Virtual Directories

By convention, tools use these virtual folders:

- `./downloads` — browser downloads
- `./screenshots` — browser screenshots
- `./outputs` — code execution images/artifacts

These are just virtual paths under the run root unless you mount them elsewhere.

## Path Mapping Example

```python
resolved = fs.resolve("./downloads/report.pdf")
resolved.virtual_path  # "/downloads/report.pdf"
resolved.host_path     # "/abs/path/to/runs/run-20260206/downloads/report.pdf"

fs.to_virtual(resolved.host_path)  # "./downloads/report.pdf"
```

## Why It Matters

With `RunFileSystem`:

- Agents can safely pass file paths to each other.
- Tool outputs are consistent across handoffs.
- You can swap local storage for sandboxed or cloud-backed storage later without changing agent contracts.

Use virtual paths everywhere in tool calls and agent prompts to keep workflows stable and portable.

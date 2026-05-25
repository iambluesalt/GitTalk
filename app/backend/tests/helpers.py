"""
Shared constants, sample source strings, and pure helper functions
used across test files. Import this directly — unlike conftest.py,
this is a regular Python module.
"""
import math
import uuid
from datetime import datetime
from pathlib import Path

EMBED_DIM = 768


def _fake_vector(seed: int = 0) -> list[float]:
    """Deterministic unit-ish embedding vector derived from a seed."""
    return [math.sin(seed * 0.37 + i * 0.13) for i in range(EMBED_DIM)]


def _make_project_meta(project_id: str, clone_path: Path, status: str = "indexed"):
    """Build a ProjectMetadata object ready for DB insertion."""
    import sys
    from pathlib import Path as P
    backend = P(__file__).resolve().parent.parent
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))

    from models import ProjectMetadata, ProjectStatus

    return ProjectMetadata(
        id=project_id,
        name="testrepo",
        github_url=f"https://github.com/test/{project_id}",
        clone_path=str(clone_path),
        status=ProjectStatus(status),
        cloned_at=datetime.now(),
    )


# ── Sample source-code strings ────────────────────────────────────────────────

SIMPLE_PY = """\
def add(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b


def subtract(a, b):
    \"\"\"Subtract b from a.\"\"\"
    return a - b
"""

CLASS_PY = """\
class Calculator:
    \"\"\"A simple calculator.\"\"\"

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(result)
        return result

    def reset(self):
        self.history = []
"""

IMPORTS_ONLY_PY = """\
import os
import sys
from pathlib import Path
from typing import Optional, List
"""

EMPTY_PY = ""

# Deliberately large — forces the chunker to split it
LARGE_FN_PY = "def big_function(x):\n" + "    x = x + 1\n" * 350 + "    return x\n"

SYNTAX_ERROR_PY = """\
def broken(:
    pass
    return 1 2 3
"""

MARKDOWN_MD = """\
# GitTalk

A tool for chatting with your codebase.

## Installation

```bash
pip install gittalk
```

## Usage

Clone a repo, then start chatting.
"""

JSON_FILE = '{"name": "gittalk", "version": "1.0.0", "description": "Chat with code"}'

SIMPLE_JS = """\
function greet(name) {
    return `Hello, ${name}!`;
}

const add = (a, b) => a + b;
"""

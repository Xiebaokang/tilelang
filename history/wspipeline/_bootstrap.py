"""Minimal runtime-path setup for importing the standalone source package."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def ensure_tvm_importable() -> None:
    """Expose the in-tree TVM only when no installed TVM is importable."""

    if importlib.util.find_spec("tvm") is not None:
        return
    repository_root = Path(__file__).resolve().parent.parent
    tvm_python = repository_root / "3rdparty" / "tvm" / "python"
    if not tvm_python.is_dir():
        return
    sys.path.insert(0, str(tvm_python))

    library_paths = (
        repository_root / "build" / "lib",
        repository_root / "build" / "tvm",
    )
    existing = os.environ.get("TVM_LIBRARY_PATH", "")
    available = [str(path) for path in library_paths if path.is_dir()]
    if existing:
        available.append(existing)
    if available:
        os.environ["TVM_LIBRARY_PATH"] = os.pathsep.join(available)

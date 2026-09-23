"""Load a kernel factory from an example without importing benchmark code."""

from __future__ import annotations

import ast
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import tilelang
import tilelang.language as T
import torch
from tvm import tirx

from OverlapPlaner.arch.hopper import HOPPER_CUDA_TARGET


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=None)
def load_example_factory(
    relative_path: str,
    function_name: str,
    helper_names: tuple[str, ...] = (),
):
    """Return an undecorated scalar-argument kernel factory from ``examples``.

    Example modules often import optional reference packages or allocate CUDA
    tensors at import time.  The tuning operators need only the DSL factory,
    so parse and execute that function plus explicitly named local helpers.
    """

    path = _REPOSITORY_ROOT / relative_path
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    wanted = {function_name, *helper_names}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    found = {node.name for node in definitions}
    missing = wanted - found
    if missing:
        raise ValueError(f"missing functions in {path}: {sorted(missing)}")
    for node in definitions:
        node.decorator_list = []
    module = ast.Module(
        body=definitions,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "T": T,
        "math": math,
        "tilelang": tilelang,
        "tirx": tirx,
        "torch": torch,
    }
    exec(compile(module, str(path), "exec", dont_inherit=True), namespace)
    return namespace[function_name]


def mark_for_overlap(prim_func: tirx.PrimFunc) -> tirx.PrimFunc:
    if not isinstance(prim_func, tirx.PrimFunc):
        raise TypeError(
            "example factory must return one TIRX PrimFunc; multi-kernel and "
            "tensor-specialized examples need a dedicated adapter"
        )
    return prim_func.with_attr("tl.auto_overlap", True)


class NativeKernelReference:
    """Use the unmodified TileLang lowering as a schedule oracle.

    Several research examples have no small standalone PyTorch reference for
    the selected internal kernel.  Comparing identical inputs against native
    lowering still detects schedule/lowering changes while keeping the example
    operator self-contained.
    """

    def __init__(self, prim_func, out_idx, pass_configs=None):
        self.prim_func = prim_func.without_attr("tl.auto_overlap")
        self.out_idx = tuple(out_idx)
        self.pass_configs = dict(pass_configs or {})
        self.compiled = None

    def __call__(self, *inputs):
        if self.compiled is None:
            self.compiled = tilelang.compile(
                self.prim_func,
                out_idx=list(self.out_idx),
                target=HOPPER_CUDA_TARGET,
                execution_backend="cython",
                pass_configs=self.pass_configs,
            )
        return self.compiled(*inputs)

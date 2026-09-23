import math

import pytest
import tilelang
import torch
import history.wspipeline as wsp
from tvm import IRModule, tirx
from tvm.error import TVMError
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit

from history.wspipeline.test.fa3_kernel import flashattn
from history.wspipeline.integration.apply import apply_schedule_to_ir


SCHEDULE_CASES = (
    "baseline_256_threads",
    "pipelined_versions_a",
    "two_group_256_threads",
    "two_group_versions_b_256_threads",
    "two_group_512_threads",
    "register_versions_a_384_threads",
    "register_versions_b_384_threads",
)


def _target():
    return Target(
        {
            "kind": "cuda",
            "arch": "sm_90a",
            "max_threads_per_block": 1024,
            "max_shared_memory_per_block": 232448,
        }
    )


def _prim_func():
    return flashattn.jit_impl.get_tir(
        1,
        1,
        256,
        256,
        64,
        False,
        block_M=128,
        block_N=128,
        threads=256,
        auto_wsp=True,
    )


def _prepare_analysis_input(target):
    mod = IRModule({"main": _prim_func()})
    for transform in (
        tirx.transform.BindTarget(target),
        tilelang.transform.MaterializeKernelLaunch(),
        tilelang.transform.AddWrapperForSingleBufStore(),
        tilelang.transform.LegalizeNegativeIndex(),
        tilelang.transform.InjectAssumes(),
        tilelang.transform.Simplify(),
        tilelang.transform.LayoutReducer(),
    ):
        mod = transform(mod)
    return mod


def _schedule_batch(analysis, target):
    selected = {}
    two_group_version_keys = []
    register_version_keys = []
    for candidate in wsp.enumerate_realized_schedules(
        analysis, target, examined_limit=300
    ):
        groups = candidate.logical.partition.num_groups
        stages = max(
            schedule.num_stages for schedule in candidate.logical.region_schedules
        )
        version_key = tuple(
            buffer.version_count for buffer in candidate.buffers.buffers
        )
        versions = max(version_key)
        threads = candidate.warp_allocation.effective_threads

        if groups == 1 and stages == 1 and threads == 256:
            selected.setdefault("baseline_256_threads", candidate)
        if groups == 1 and stages == 2 and versions > 1 and threads == 256:
            selected.setdefault("pipelined_versions_a", candidate)
        if (
            groups == 2
            and candidate.channels
            and threads == 256
            and version_key[13] == 2
        ):
            if version_key not in two_group_version_keys:
                two_group_version_keys.append(version_key)
                if len(two_group_version_keys) == 1:
                    selected["two_group_256_threads"] = candidate
                elif len(two_group_version_keys) == 2:
                    selected["two_group_versions_b_256_threads"] = candidate
        if groups == 2 and candidate.channels and threads == 512:
            selected.setdefault("two_group_512_threads", candidate)
        if (
            groups == 3
            and stages == 2
            and version_key[13] == 2
            and all(
                version == 1 or buffer_id in {6, 13}
                for buffer_id, version in enumerate(version_key)
            )
            and candidate.register_allocation.setmaxnreg_enabled
            and threads == 384
        ):
            if version_key not in register_version_keys:
                register_version_keys.append(version_key)
                if len(register_version_keys) <= 2:
                    suffix = "ab"[len(register_version_keys) - 1]
                    selected[f"register_versions_{suffix}_384_threads"] = candidate
        if len(selected) == len(SCHEDULE_CASES):
            return selected

    raise AssertionError(f"missing schedule cases: {set(SCHEDULE_CASES) - set(selected)}")


def _function(mod):
    return mod[mod.get_global_var("main")]


def _program_metadata(func):
    attr_keys = []
    loop_annotation_keys = []

    def collect(node):
        if isinstance(node, tirx.AttrStmt):
            attr_keys.append(node.attr_key)
        if isinstance(node, tirx.For):
            loop_annotation_keys.extend(str(key) for key in node.annotations)

    post_order_visit(func.body, collect)
    return attr_keys, loop_annotation_keys


def _assert_completion_attrs(func):
    for name in (
        "groups_lowered",
        "pipeline_schedules_lowered",
        "buffers_lowered",
        "synchronization_lowered",
    ):
        assert int(func.attrs[f"tl.program_schedule.{name}"]) == 1


def test_cpp_passes_accept_schedule_batch():
    """Run every WSP-related C++ IR pass over representative schedules."""

    target = _target()
    analysis_input = _prepare_analysis_input(target)
    analysis = wsp.analyze_program_dataflow(_function(analysis_input))

    for case, candidate in _schedule_batch(analysis, target).items():
        scheduled = apply_schedule_to_ir(
            analysis_input,
            analysis,
            candidate,
            target,
            global_symbol="main",
        )
        assert int(_function(scheduled).attrs["tl.program_schedule.version"]) == 3

        if case == "baseline_256_threads":
            with pytest.raises(TVMError, match="cannot consume a program-aware"):
                tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(scheduled)

        lowered = tilelang.transform.LowerProgramSchedule()(scheduled)
        lowered_func = _function(lowered)
        _assert_completion_attrs(lowered_func)
        attr_keys, _ = _program_metadata(lowered_func)
        assert "tl.program_schedule.group_scope" in attr_keys, case
        assert "tl.program_schedule.operation" in attr_keys, case

        lowered = tilelang.cuda.transform.LowerBlackwell2SM()(lowered)
        lowered = tilelang.transform.IfStmtBinding()(lowered)
        lowered = tilelang.transform.PipelinePlanning()(lowered)
        lowered = tilelang.transform.InjectSoftwarePipeline()(lowered)
        lowered = tilelang.transform.Simplify()(lowered)
        with target:
            lowered = tilelang.transform.LayoutInference()(lowered)
            lowered = tilelang.transform.LowerTileOp()(lowered)

        lowered_func = _function(lowered)
        _assert_completion_attrs(lowered_func)
        assert int(
            lowered_func.attrs["tl.program_schedule.tile_ops_lowered"]
        ) == 1

        finalized = tilelang.transform.FinalizeProgramSchedule()(lowered)
        finalized_func = _function(finalized)
        assert int(finalized_func.attrs["tl.program_schedule.lowered"]) == 3
        assert "tl.program_schedule.version" not in finalized_func.attrs

        attr_keys, loop_annotation_keys = _program_metadata(finalized_func)
        assert not any(key.startswith("tl.program_schedule") for key in attr_keys)
        assert not any(
            key.startswith("tl.program_schedule") or key == "tl.wsp.auto_schedule"
            for key in loop_annotation_keys
        )


def _has_hopper_gpu():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.parametrize("case", SCHEDULE_CASES)
@pytest.mark.skipif(not _has_hopper_gpu(), reason="requires a Hopper CUDA GPU")
def test_schedule_compiles_and_is_correct(case):
    """Compile and execute one representative schedule through the full pipeline."""

    target = _target()
    torch.manual_seed(0)
    q = torch.randn(1, 1, 256, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(64)
    expected = (scores.softmax(dim=-1) @ v.float()).half()

    selected = {}

    def planner(_name, analysis, planner_target):
        candidate = _schedule_batch(analysis, planner_target)[case]
        selected[case] = candidate
        return candidate

    prim_func = _prim_func().with_attr(
        "tl.program_schedule.request", f"cpp-pass-test={case}"
    )
    with wsp.use_schedule_planner(planner):
        kernel = tilelang.compile(
            prim_func,
            out_idx=[3],
            target=target,
            execution_backend="cython",
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )

    assert case in selected
    actual = kernel(q, k, v)
    torch.testing.assert_close(
        actual, expected, rtol=0.01, atol=0.01, msg=f"schedule case: {case}"
    )

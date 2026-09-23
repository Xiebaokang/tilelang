from tilelang import tvm as tvm
import tilelang
from tvm.script import tirx as T
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


@T.prim_func
def _pingpong(A: T.Buffer((8,), "float32"), B: T.Buffer((8,), "float32")):
    acc0 = T.alloc_buffer((1,), "float32", scope="local")
    acc1 = T.alloc_buffer((1,), "float32", scope="local")
    for k in range(8):
        if k % 2 == 0:
            acc0[0] = A[k]
            B[k] = acc0[0]
        else:
            acc1[0] = A[k]
            B[k] = acc1[0]


def _pipeline_loops(stmt):
    loops = []

    def visit(node):
        if isinstance(node, tirx.For):
            loops.append(node)

    post_order_visit(stmt, visit)
    return loops


def test_unroll_fragment_pingpong_expands_mod_two_loop() -> None:
    mod = tvm.IRModule({"main": _pingpong})
    after = tilelang.transform.UnrollFragmentPingPong()(mod)["main"]
    loops = _pipeline_loops(after.body)
    assert len(loops) == 1
    assert int(loops[0].extent) == 4
    copies = loops[0].body
    assert isinstance(copies, tirx.SeqStmt)
    assert len(copies) == 2

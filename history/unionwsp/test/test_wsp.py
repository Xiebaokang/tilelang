"""Count automatically inferred UnionWSP schedules for FA3."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "3rdparty" / "tvm" / "python"))
os.environ.setdefault("TVM_LIBRARY_PATH", str(PROJECT_ROOT / "build" / "lib"))

from history.unionwsp import enumerate_wsp_schedules
from history.unionwsp.parseIR import extract_dataflow_graph
from history.unionwsp.test.test_stage import hopper_target, make_fa3_prim_func


def count_fa3_automatic_schedules() -> int:
    graph = extract_dataflow_graph(
        make_fa3_prim_func(), target=hopper_target()
    )
    count = sum(
        1
        for _ in enumerate_wsp_schedules(
            graph,
            original_threads=256,
        )
    )
    print(f"FA3 automatic WSP schedules: {count}")
    return count


if __name__ == "__main__":
    count_fa3_automatic_schedules()

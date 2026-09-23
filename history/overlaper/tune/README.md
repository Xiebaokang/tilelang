# Running an Overlaper search

Choose operators by importing their `OPERATOR` objects in `run.py` and editing
the list near the top of that file:

```python
from overlaper.tune.operators.fa3 import OPERATOR as FA3
from overlaper.tune.operators.gemm import OPERATOR as GEMM
from overlaper.tune.operators.gqa import OPERATOR as GQA
from overlaper.tune.operators.gqa_bwd import OPERATOR as GQA_BWD
from overlaper.tune.operators.mha_bwd import OPERATOR as MHA_BWD
from overlaper.tune.operators.mamba_scan import OPERATOR as MAMBA_SCAN
from overlaper.tune.operators.mla import OPERATOR as MLA

SEARCH_OPERATORS = [FA3, GEMM, GQA, MHA_BWD, GQA_BWD, MAMBA_SCAN, MLA]
```

Then run:

```bash
PYTHONPATH="$PWD/3rdparty/tvm/python:$PWD" \
TVM_LIBRARY_PATH="$PWD/build/lib" \
python -u -m overlaper.tune.run \
  --compile-timeout 120 \
  --execution-timeout 30 \
  --output overlaper_tune_results
```

For a smoke test, add `--max-schedules-per-config 10 --warmup 2 --rep 5`.
Each selected operator is searched over every tile returned by its
`configurations()` function. Results are ranked independently in
`<output>/<operator>/top30.json`; failed or timed-out schedules remain below
their tile directory.

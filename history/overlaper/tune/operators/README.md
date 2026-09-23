# Operator interface

Every operator module exposes exactly one public registration object named
`OPERATOR`. The module implements three small functions and binds them to an
`OperatorSpec`:

```python
from .workloads import OperatorSpec, SearchWorkload


def add_arguments(parser) -> None:
    # Add arguments with an operator-specific prefix.
    ...


def configurations(options) -> list[dict[str, int]]:
    # Return the Cartesian tile search space.
    ...


def build(options, config) -> SearchWorkload:
    # Materialize exactly one tile as a TIRX PrimFunc and its benchmark data.
    return SearchWorkload(
        prim_func=...,
        out_idx=(...),
        total_flops=...,
        reference_program=...,
        input_tensors=None,
        pass_configs={},
    )


OPERATOR = OperatorSpec(
    name="operator_name",
    description="Short description",
    add_cli_arguments=add_arguments,
    configuration_factory=configurations,
    workload_factory=build,
)
```

Then import `OPERATOR` in `operators/__init__.py` and append it to `OPERATORS`.
The registry validates operator names, tile keys, positive tile values,
duplicate configurations, PrimFunc type, output indices, FLOPs, and reference
callability before the measured search begins.

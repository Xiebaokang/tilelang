"""Triton baseline for the MHA backward operator."""
try:
    from .attention_bwd import run_operator
except ImportError:
    from attention_bwd import run_operator


def run(*, warmup=100, rep=400, trials=5):
    return run_operator("mha_bwd", groups=1, warmup=warmup, rep=rep, trials=trials)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: diagonal tensor must be rank >= 1 and last dimension > 0

rule_1 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["diag_ndim"] >= 1,
                Select(v["diag_shape"], v["diag_ndim"] - 1) > 0
            )
        )
    ) if n else
    s.add(
        And(
            v["diag_ndim"] >= 1,
            Select(v["diag_shape"], v["diag_ndim"] - 1) > 0
        )
    )
)
def rule_1_func(arg1, solver=None, neg=False):
    diagonal = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(diagonal, np.ndarray):
            return False

        solver = Solver()

        diag_ndim = Int('diag_ndim')
        diag_shape = Array('diag_shape', IntSort(), IntSort())

        solver.add(diag_ndim == diagonal.ndim)
        for i in range(diagonal.ndim):
            diag_shape = Store(diag_shape, i, diagonal.shape[i])

        rule_1(
            solver,
            {
                "diag_ndim": diag_ndim,
                "diag_shape": diag_shape
            }
        )

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1(
            solver,
            {
                "diag_ndim": diagonal["ndim"],
                "diag_shape": diagonal["shape"]
            },
            neg
        )
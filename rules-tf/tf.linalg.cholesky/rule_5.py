import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
"""
1. Input must be rank ≥ 2
2. The last two dimensions must form a square matrix
3. The matrix dimensions must be strictly positive
"""
rule_5 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["input_ndim"] >= 2,
                Select(v["input_shape"], v["input_ndim"] - 1) > 0,
                Select(v["input_shape"], v["input_ndim"] - 2) > 0,
                Select(v["input_shape"], v["input_ndim"] - 1) ==
                Select(v["input_shape"], v["input_ndim"] - 2)
            )
        )
    ) if n else
    s.add(
        And(
            v["input_ndim"] >= 2,
            Select(v["input_shape"], v["input_ndim"] - 1) > 0,
            Select(v["input_shape"], v["input_ndim"] - 2) > 0,
            Select(v["input_shape"], v["input_ndim"] - 1) ==
            Select(v["input_shape"], v["input_ndim"] - 2)
        )
    )
)
def rule_5_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        solver = Solver()

        input_ndim = Int('input_ndim')
        input_shape = Array('input_shape', IntSort(), IntSort())

        solver.add(input_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            input_shape = Store(input_shape, i, arg1.shape[i])

        rule_5(
            solver,
            {
                "input_ndim": input_ndim,
                "input_shape": input_shape
            }
        )

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(
            solver,
            {
                "input_ndim": arg1["ndim"],
                "input_shape": arg1["shape"]
            },
            neg
        )
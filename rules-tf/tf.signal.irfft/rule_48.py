import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 48: fft_length is None
# Input must have rank >= 1 and last dimension >= 1

rule_48 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["input_ndim"] >= 1,
                And([Select(v["input_shape"], i) > 0 for i in range(MAX_N_DIM)]),
                Select(v["input_shape"], v["input_ndim"] - 1) >= 1
            )
        )
    ) if n else
    s.add(
        And(
            v["input_ndim"] >= 1,
            And([Implies(i < v["input_ndim"],
                         Select(v["input_shape"], i) > 0)
                 for i in range(MAX_N_DIM)]),
            Select(v["input_shape"], v["input_ndim"] - 1) >= 1
        )
    )
)
def rule_48_func(arg1, solver=None, neg=False):
    input_tensor = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(input_tensor, np.ndarray):
            return False

        solver = Solver()
        input_ndim = Int('input_ndim')
        input_shape = Array('input_shape', IntSort(), IntSort())

        solver.add(input_ndim == input_tensor.ndim)

        for i in range(input_tensor.ndim):
            solver.add(Select(input_shape, i) == input_tensor.shape[i])

        rule_48(solver, {
            "input_ndim": input_ndim,
            "input_shape": input_shape
        })

        return solver.check() == sat

    # Fuzz generation phase
    else:
        rule_48(
            solver,
            {
                "input_ndim": input_tensor["ndim"],
                "input_shape": input_tensor["shape"]
            },
            neg
        )
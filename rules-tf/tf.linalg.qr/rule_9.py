import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: Valid input shape for tf.linalg.qr with full_matrices=True
"""
1. Rank ≥ 2
2. Last two dimensions strictly positive:
3. All dimensions > 0
"""
rule_9 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # rank >= 2
                v["input_ndim"] >= 2,

                # M > 0
                Select(v["input_shape"], v["input_ndim"] - 2) > 0,

                # N > 0
                Select(v["input_shape"], v["input_ndim"] - 1) > 0
            )
        )
    ) if n else
    s.add(
        And(
            v["input_ndim"] >= 2,
            Select(v["input_shape"], v["input_ndim"] - 2) > 0,
            Select(v["input_shape"], v["input_ndim"] - 1) > 0
        )
    )
)

def rule_9_func(arg1, solver=None, neg=False):

    input_tensor = next(iter(arg1.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(input_tensor, np.ndarray):
            return False

        solver = Solver()

        # Declare symbolic variables
        input_ndim = Int("input_ndim")
        input_shape = Array("input_shape", IntSort(), IntSort())

        # Assign concrete values
        solver.add(input_ndim == input_tensor.ndim)

        for i in range(input_tensor.ndim):
            input_shape = Store(input_shape, i, input_tensor.shape[i])

        # Apply constraint
        rule_9(
            solver,
            {
                "input_ndim": input_ndim,
                "input_shape": input_shape,
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz input generation phase
    # -----------------------------
    else:

        rule_9(
            solver,
            {
                "input_ndim": input_tensor["ndim"],
                "input_shape": input_tensor["shape"],
            },
            neg
        )
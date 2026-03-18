import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: Valid input for tf.math.argmax with default axis=0

rule_116 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # rank >= 1
                v["input_ndim"] >= 1,

                # dimension 0 must exist and be > 0
                Select(v["input_shape"], 0) > 0
            )
        )
    ) if n else
    s.add(
        And(
            v["input_ndim"] >= 1,
            Select(v["input_shape"], 0) > 0
        )
    )
)

def rule_116_func(arg1, solver=None, neg=False):

    input_tensor = next(iter(arg1.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(input_tensor, np.ndarray):
            return False

        solver = Solver()

        # Symbolic variables
        input_ndim = Int("input_ndim")
        input_shape = Array("input_shape", IntSort(), IntSort())

        # Assign concrete values
        solver.add(input_ndim == input_tensor.ndim)

        for i in range(input_tensor.ndim):
            input_shape = Store(input_shape, i, input_tensor.shape[i])

        # Apply constraint
        rule_116(
            solver,
            {
                "input_ndim": input_ndim,
                "input_shape": input_shape,
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz generation phase
    # -----------------------------
    else:

        rule_116(
            solver,
            {
                "input_ndim": input_tensor["ndim"],
                "input_shape": input_tensor["shape"],
            },
            neg
        )
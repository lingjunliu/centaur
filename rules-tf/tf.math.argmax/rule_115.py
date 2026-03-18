import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: Valid input for tf.math.argmax
"""
1. rank ≥ 1
2. valid axis
3. selected dimension > 0
"""

rule_115 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # rank >= 1
                v["input_ndim"] >= 1,

                # axis valid (supports negative axis)
                v["axis"] >= -v["input_ndim"],
                v["axis"] < v["input_ndim"],

                # dimension along axis > 0
                Select(
                    v["input_shape"],
                    If(v["axis"] >= 0,
                       v["axis"],
                       v["axis"] + v["input_ndim"])
                ) > 0
            )
        )
    ) if n else
    s.add(
        And(
            v["input_ndim"] >= 1,
            v["axis"] >= -v["input_ndim"],
            v["axis"] < v["input_ndim"],
            Select(
                v["input_shape"],
                If(v["axis"] >= 0,
                   v["axis"],
                   v["axis"] + v["input_ndim"])
            ) > 0
        )
    )
)

def rule_115_func(arg1, arg2, solver=None, neg=False):

    input_tensor = next(iter(arg1.values()))
    axis = next(iter(arg2.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(input_tensor, np.ndarray):
            return False

        solver = Solver()

        # Symbolic vars
        input_ndim = Int("input_ndim")
        input_shape = Array("input_shape", IntSort(), IntSort())
        axis_sym = Int("axis")

        # Assign concrete values
        solver.add(input_ndim == input_tensor.ndim)
        solver.add(axis_sym == axis)

        for i in range(input_tensor.ndim):
            input_shape = Store(input_shape, i, input_tensor.shape[i])

        # Apply rule
        rule_115(
            solver,
            {
                "input_ndim": input_ndim,
                "input_shape": input_shape,
                "axis": axis_sym,
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz generation phase
    # -----------------------------
    else:

        rule_115(
            solver,
            {
                "input_ndim": input_tensor["ndim"],
                "input_shape": input_tensor["shape"],
                "axis": axis,
            },
            neg
        )
import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: Valid input for tf.math.cumprod with arbitrary valid axis

rule_127 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # rank >= 1
                v["x_ndim"] >= 1,

                # axis in valid range
                v["axis"] >= -v["x_ndim"],
                v["axis"] < v["x_ndim"],

                # dimension along axis > 0
                Select(
                    v["x_shape"],
                    If(v["axis"] >= 0,
                       v["axis"],
                       v["axis"] + v["x_ndim"])
                ) > 0
            )
        )
    ) if n else
    s.add(
        And(
            v["x_ndim"] >= 1,
            v["axis"] >= -v["x_ndim"],
            v["axis"] < v["x_ndim"],
            Select(
                v["x_shape"],
                If(v["axis"] >= 0,
                   v["axis"],
                   v["axis"] + v["x_ndim"])
            ) > 0
        )
    )
)

def rule_127_func(arg1, arg2, solver=None, neg=False):

    x = next(iter(arg1.values()))
    axis = next(iter(arg2.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(x, np.ndarray):
            return False

        solver = Solver()

        # Symbolic variables
        x_ndim = Int("x_ndim")
        x_shape = Array("x_shape", IntSort(), IntSort())
        axis_sym = Int("axis")

        # Concrete assignments
        solver.add(x_ndim == x.ndim)
        solver.add(axis_sym == axis)

        for i in range(x.ndim):
            x_shape = Store(x_shape, i, x.shape[i])

        # Apply constraint
        rule_127(
            solver,
            {
                "x_ndim": x_ndim,
                "x_shape": x_shape,
                "axis": axis_sym,
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz generation phase
    # -----------------------------
    else:

        rule_127(
            solver,
            {
                "x_ndim": x["ndim"],
                "x_shape": x["shape"],
                "axis": axis,
            },
            neg
        )
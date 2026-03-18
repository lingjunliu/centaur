import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: axis is within [-ndim, ndim-1] and selected dimension size > 0

rule_77 = lambda s, v, n=False: (
    s.add(Not(And(
        v["arg1_ndim"] > 0,
        v["axis"] >= -v["arg1_ndim"],
        v["axis"] < v["arg1_ndim"],
        Select(
            v["arg1_shape"],
            If(v["axis"] < 0, v["axis"] + v["arg1_ndim"], v["axis"])
        ) > 0
    ))) if n else
    And(
        v["arg1_ndim"] > 0,
        v["axis"] >= -v["arg1_ndim"],
        v["axis"] < v["arg1_ndim"],
        Select(
            v["arg1_shape"],
            If(v["axis"] < 0, v["axis"] + v["arg1_ndim"], v["axis"])
        ) > 0
    )
)
def rule_77_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    axis = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(axis, int):
            return False

        solver = Solver()

        # Z3 variables
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        axis_z3 = Int('axis')

        # Assign values
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(axis_z3 == axis)

        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Apply rule
        rule_77(
            solver,
            {
                "arg1_ndim": arg1_ndim,
                "arg1_shape": arg1_shape,
                "axis": axis_z3
            }
        )

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(
            solver,
            {
                "arg1_ndim": arg1["ndim"],
                "arg1_shape": arg1["shape"],
                "axis": axis
            },
            neg
        )
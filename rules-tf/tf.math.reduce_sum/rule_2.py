import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 2: valid axis provided

rule_2 = lambda s, v, n=False: (
    s.add(Not(And(

        v["arg1_ndim"] >= 1,

        # tensor dimensions valid
        And([
            Implies(i < v["arg1_ndim"],
                    Select(v["arg1_shape"], i) > 0)
            for i in range(MAX_N_DIM)
        ]),

        # axis validity constraint
        v["axis"] >= -v["arg1_ndim"],
        v["axis"] < v["arg1_ndim"]

    )) if n else
    And(

        v["arg1_ndim"] >= 1,

        And([
            Implies(i < v["arg1_ndim"],
                    Select(v["arg1_shape"], i) > 0)
            for i in range(MAX_N_DIM)
        ]),

        v["axis"] >= -v["arg1_ndim"],
        v["axis"] < v["arg1_ndim"]
    ))
)
def rule_2_func(arg1, arg2, solver=None, neg=False):

    arg1 = next(iter(arg1.values()))
    axis = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:

        if not isinstance(arg1, np.ndarray):
            return False

        if not isinstance(axis, int):
            return False

        solver = Solver()

        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        axis_sym = Int('axis')

        solver.add(arg1_ndim == arg1.ndim)
        solver.add(axis_sym == axis)

        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        rule_2(
            solver,
            {
                "arg1_ndim": arg1_ndim,
                "arg1_shape": arg1_shape,
                "axis": axis_sym
            }
        )

        return solver.check() == sat

    # Fuzz input generation phase
    else:

        rule_2(
            solver,
            {
                "arg1_ndim": arg1["ndim"],
                "arg1_shape": arg1["shape"],
                "axis": axis
            },
            neg
        )
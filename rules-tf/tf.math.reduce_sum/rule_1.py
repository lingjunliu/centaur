import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 1: axis=None and keepdims=False

rule_1 = lambda s, v, n=False: (
    s.add(Not(And(
        v["arg1_ndim"] >= 1,

        # each active dimension must be > 0
        And([
            Implies(i < v["arg1_ndim"],
                    Select(v["arg1_shape"], i) > 0)
            for i in range(MAX_N_DIM)
        ])
    )) if n else
    And(
        v["arg1_ndim"] >= 1,

        And([
            Implies(i < v["arg1_ndim"],
                    Select(v["arg1_shape"], i) > 0)
            for i in range(MAX_N_DIM)
        ])
    ))
)
def rule_1_func(arg1, solver=None, neg=False):

    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:

        if not isinstance(arg1, np.ndarray):
            return False

        solver = Solver()

        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        solver.add(arg1_ndim == arg1.ndim)

        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        rule_1(
            solver,
            {
                "arg1_ndim": arg1_ndim,
                "arg1_shape": arg1_shape
            }
        )

        return solver.check() == sat

    # Fuzz generation phase
    else:

        rule_1(
            solver,
            {
                "arg1_ndim": arg1["ndim"],
                "arg1_shape": arg1["shape"]
            },
            neg
        )
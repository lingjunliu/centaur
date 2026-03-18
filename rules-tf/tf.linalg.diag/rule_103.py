import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: diagonal tensor must be rank >= 2, last dim > 0,
# and second-to-last dim matches number of diagonals implied by k

rule_103 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["diag_ndim"] >= 2,
                Select(v["diag_shape"], v["diag_ndim"] - 1) > 0,
                Select(v["diag_shape"], v["diag_ndim"] - 2) ==
                (v["k_upper"] - v["k_lower"] + 1),
                v["k_lower"] <= v["k_upper"]
            )
        )
    ) if n else
    s.add(
        And(
            v["diag_ndim"] >= 2,
            Select(v["diag_shape"], v["diag_ndim"] - 1) > 0,
            Select(v["diag_shape"], v["diag_ndim"] - 2) ==
            (v["k_upper"] - v["k_lower"] + 1),
            v["k_lower"] <= v["k_upper"]
        )
    )
)

def rule_103_func(arg1, arg2, solver=None, neg=False):
    diagonal = next(iter(arg1.values()))
    k_lower, k_upper = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(diagonal, np.ndarray):
            return False
        if not isinstance(k_lower, (int, np.integer)):
            return False
        if not isinstance(k_upper, (int, np.integer)):
            return False
        if k_lower > k_upper:
            return False

        solver = Solver()

        diag_ndim = Int('diag_ndim')
        diag_shape = Array('diag_shape', IntSort(), IntSort())
        k_lower_z3 = Int('k_lower')
        k_upper_z3 = Int('k_upper')

        solver.add(diag_ndim == diagonal.ndim)
        for i in range(diagonal.ndim):
            diag_shape = Store(diag_shape, i, diagonal.shape[i])

        solver.add(k_lower_z3 == int(k_lower))
        solver.add(k_upper_z3 == int(k_upper))

        rule_103(
            solver,
            {
                "diag_ndim": diag_ndim,
                "diag_shape": diag_shape,
                "k_lower": k_lower_z3,
                "k_upper": k_upper_z3
            }
        )

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(
            solver,
            {
                "diag_ndim": diagonal["ndim"],
                "diag_shape": diagonal["shape"],
                "k_lower": k_lower,
                "k_upper": k_upper
            },
            neg
        )
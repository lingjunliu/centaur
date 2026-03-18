import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: input tensor must be rank >= 2, last two dims > 0,
# and k must define a valid diagonal range

rule_6 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["input_ndim"] >= 2,
                Select(v["input_shape"], v["input_ndim"] - 1) > 0,
                Select(v["input_shape"], v["input_ndim"] - 2) > 0,
                v["k_lower"] <= v["k_upper"]
            )
        )
    ) if n else
    s.add(
        And(
            v["input_ndim"] >= 2,
            Select(v["input_shape"], v["input_ndim"] - 1) > 0,
            Select(v["input_shape"], v["input_ndim"] - 2) > 0,
            v["k_lower"] <= v["k_upper"]
        )
    )
)

def rule_6_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    k_lower, k_upper = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(k_lower, (int, np.integer)):
            return False
        if not isinstance(k_upper, (int, np.integer)):
            return False
        if k_lower > k_upper:
            return False

        solver = Solver()

        input_ndim = Int('input_ndim')
        input_shape = Array('input_shape', IntSort(), IntSort())
        k_lower_z3 = Int('k_lower')
        k_upper_z3 = Int('k_upper')

        solver.add(input_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            input_shape = Store(input_shape, i, arg1.shape[i])

        solver.add(k_lower_z3 == int(k_lower))
        solver.add(k_upper_z3 == int(k_upper))

        rule_6(
            solver,
            {
                "input_ndim": input_ndim,
                "input_shape": input_shape,
                "k_lower": k_lower_z3,
                "k_upper": k_upper_z3
            }
        )

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(
            solver,
            {
                "input_ndim": arg1["ndim"],
                "input_shape": arg1["shape"],
                "k_lower": k_lower,
                "k_upper": k_upper
            },
            neg
        )
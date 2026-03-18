import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: input tensor must be rank >= 2 and its last two dimensions > 0
# when valid num_lower and num_upper are provided

rule_5 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["input_ndim"] >= 2,
                Select(v["input_shape"], v["input_ndim"] - 1) > 0,
                Select(v["input_shape"], v["input_ndim"] - 2) > 0,
                v["num_lower"] >= -1,
                v["num_upper"] >= -1
            )
        )
    ) if n else
    s.add(
        And(
            v["input_ndim"] >= 2,
            Select(v["input_shape"], v["input_ndim"] - 1) > 0,
            Select(v["input_shape"], v["input_ndim"] - 2) > 0,
            v["num_lower"] >= -1,
            v["num_upper"] >= -1
        )
    )
)

def rule_5_func(arg1, arg2, arg3, solver=None, neg=False):
    input_tensor = next(iter(arg1.values()))
    num_lower = next(iter(arg2.values()))
    num_upper = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(input_tensor, np.ndarray):
            return False
        if not isinstance(num_lower, (int, np.integer)):
            return False
        if not isinstance(num_upper, (int, np.integer)):
            return False
        if num_lower < -1 or num_upper < -1:
            return False

        solver = Solver()

        input_ndim = Int('input_ndim')
        input_shape = Array('input_shape', IntSort(), IntSort())
        num_lower_z3 = Int('num_lower')
        num_upper_z3 = Int('num_upper')

        solver.add(input_ndim == input_tensor.ndim)
        for i in range(input_tensor.ndim):
            input_shape = Store(input_shape, i, input_tensor.shape[i])

        solver.add(num_lower_z3 == int(num_lower))
        solver.add(num_upper_z3 == int(num_upper))

        rule_5(
            solver,
            {
                "input_ndim": input_ndim,
                "input_shape": input_shape,
                "num_lower": num_lower_z3,
                "num_upper": num_upper_z3
            }
        )

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(
            solver,
            {
                "input_ndim": input_tensor["ndim"],
                "input_shape": input_tensor["shape"],
                "num_lower": num_lower,
                "num_upper": num_upper
            },
            neg
        )
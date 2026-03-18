import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: constraints for tf.linalg.band_part

rule_4 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # input rank ≥ 2
                v["input_ndim"] >= 2,

                # matrix dims positive
                Select(v["input_shape"], v["input_ndim"] - 1) > 0,
                Select(v["input_shape"], v["input_ndim"] - 2) > 0,

                # num_lower and num_upper are scalars
                v["num_lower_ndim"] == 0,
                v["num_upper_ndim"] == 0,

                # band values valid
                v["num_lower_val"] >= -1,
                v["num_upper_val"] >= -1
            )
        ) if n else
        And(
            v["input_ndim"] >= 2,

            Select(v["input_shape"], v["input_ndim"] - 1) > 0,
            Select(v["input_shape"], v["input_ndim"] - 2) > 0,

            v["num_lower_ndim"] == 0,
            v["num_upper_ndim"] == 0,

            v["num_lower_val"] >= -1,
            v["num_upper_val"] >= -1
        )
    )
)

def rule_4_func(arg1, arg2, arg3, solver=None, neg=False):

    input_tensor = next(iter(arg1.values()))
    num_lower = next(iter(arg2.values()))
    num_upper = next(iter(arg3.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(input_tensor, np.ndarray):
            return False
        if not isinstance(num_lower, np.ndarray):
            return False
        if not isinstance(num_upper, np.ndarray):
            return False

        solver = Solver()

        # Symbolic variables
        input_ndim = Int('input_ndim')
        input_shape = Array('input_shape', IntSort(), IntSort())

        num_lower_ndim = Int('num_lower_ndim')
        num_upper_ndim = Int('num_upper_ndim')

        num_lower_val = Int('num_lower_val')
        num_upper_val = Int('num_upper_val')

        # Assign input rank
        solver.add(input_ndim == input_tensor.ndim)

        for i in range(input_tensor.ndim):
            input_shape = Store(input_shape, i, input_tensor.shape[i])

        # Assign scalar ranks
        solver.add(num_lower_ndim == num_lower.ndim)
        solver.add(num_upper_ndim == num_upper.ndim)

        # Extract scalar values
        if num_lower.ndim == 0:
            solver.add(num_lower_val == int(num_lower))
        else:
            return False

        if num_upper.ndim == 0:
            solver.add(num_upper_val == int(num_upper))
        else:
            return False

        rule_4(
            solver,
            {
                "input_ndim": input_ndim,
                "input_shape": input_shape,
                "num_lower_ndim": num_lower_ndim,
                "num_upper_ndim": num_upper_ndim,
                "num_lower_val": num_lower_val,
                "num_upper_val": num_upper_val
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz generation phase
    # -----------------------------
    else:
        rule_4(
            solver,
            {
                "input_ndim": inputs["input"]["ndim"],
                "input_shape": inputs["input"]["shape"],
                "num_lower_ndim": inputs["num_lower"]["ndim"],
                "num_upper_ndim": inputs["num_upper"]["ndim"],
                "num_lower_val": inputs["num_lower"]["value"],
                "num_upper_val": inputs["num_upper"]["value"]
            },
            neg
        )
import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
rule_52 = lambda s, v, n=False: (
    s.add(Not(
        And(
            # axis must be within valid range
            v["axis_value"] >= -v["input_ndim"],
            v["axis_value"] < v["input_ndim"],

            # input must have at least 1 dim if axis used
            v["input_ndim"] > 0
        )
    )) if n else
    s.add(
        And(
            v["axis_value"] >= -v["input_ndim"],
            v["axis_value"] < v["input_ndim"],
            v["input_ndim"] > 0
        )
    )
)
def rule_52_func(arg1, arg2, solver=None, neg=False):

    input_tensor = next(iter(arg1.values()))
    axis_value = next(iter(arg2.values()))

    if not solver:
        if not isinstance(input_tensor, np.ndarray):
            return False

        solver = Solver()

        input_ndim = Int('input_ndim')
        axis_val = Int('axis_value')

        solver.add(input_ndim == input_tensor.ndim)
        solver.add(axis_val == axis_value)

        rule_52(solver, {
            "input_ndim": input_ndim,
            "axis_value": axis_val
        })

        return solver.check() == sat

    else:
        rule_52(solver, {
            "input_ndim": input_tensor["ndim"],
            "axis_value": axis_value
        }, neg)
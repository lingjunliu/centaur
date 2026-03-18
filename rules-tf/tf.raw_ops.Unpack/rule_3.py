import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
rule_3 = lambda s, v, n=False: (
    s.add(Not(
        And(
            v["value_ndim"] >= 1,
            v["value_ndim"] <= MAX_N_DIM,
            Select(v["value_shape"], 0) > 0,
            v["num"] == Select(v["value_shape"], 0),
            v["axis"] == 0
        )
    )) if n else
    s.add(
        And(
            v["value_ndim"] >= 1,
            v["value_ndim"] <= MAX_N_DIM,
            Select(v["value_shape"], 0) > 0,
            v["num"] == Select(v["value_shape"], 0),
            v["axis"] == 0
        )
    )
)
def rule_3_func(arg1, solver=None, neg=False):

    value = next(iter(arg1.values()))

    if not solver:
        if not isinstance(value, np.ndarray):
            return False

        solver = Solver()

        value_ndim = Int('value_ndim')
        value_shape = Array('value_shape', IntSort(), IntSort())
        num = Int('num')
        axis = Int('axis')

        solver.add(value_ndim == value.ndim)

        for i in range(value.ndim):
            value_shape = Store(value_shape, i, value.shape[i])

        solver.add(axis == 0)
        solver.add(num == value.shape[0])

        rule_3(solver, {
            "value_ndim": value_ndim,
            "value_shape": value_shape,
            "num": num,
            "axis": axis
        })

        return solver.check() == sat

    else:
        rule_3(solver, {
            "value_ndim": value["ndim"],
            "value_shape": value["shape"],
            "num": value["shape"][0],
            "axis": 0
        }, neg)
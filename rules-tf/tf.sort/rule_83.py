import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 83: valid axis AND direction

rule_83 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["values_ndim"] >= 1,
                v["direction_valid"],
                v["axis"] >= -v["values_ndim"],
                v["axis"] < v["values_ndim"],
                And([
                    Implies(i < v["values_ndim"],
                            Select(v["values_shape"], i) > 0)
                    for i in range(MAX_N_DIM)
                ])
            )
        )
    ) if n else
    s.add(
        And(
            v["values_ndim"] >= 1,
            v["direction_valid"],
            v["axis"] >= -v["values_ndim"],
            v["axis"] < v["values_ndim"],
            And([
                Implies(i < v["values_ndim"],
                        Select(v["values_shape"], i) > 0)
                for i in range(MAX_N_DIM)
            ])
        )
    )
)
def rule_83_func(arg1, arg2, arg3, solver=None, neg=False):
    values = next(iter(arg1.values()))
    axis = next(iter(arg2.values()))
    direction = next(iter(arg3.values()))

    if not solver:
        if not isinstance(values, np.ndarray):
            return False
        if direction not in ["ASCENDING", "DESCENDING"]:
            return False
        if not (-values.ndim <= axis < values.ndim):
            return False

        solver = Solver()

        values_ndim = Int('values_ndim')
        values_shape = Array('values_shape', IntSort(), IntSort())
        axis_var = Int('axis')
        direction_valid = Bool('direction_valid')

        solver.add(values_ndim == values.ndim)
        solver.add(axis_var == axis)
        solver.add(direction_valid == True)

        for i in range(values.ndim):
            solver.add(Select(values_shape, i) == values.shape[i])

        rule_83(solver, {
            "values_ndim": values_ndim,
            "values_shape": values_shape,
            "axis": axis_var,
            "direction_valid": direction_valid
        })

        return solver.check() == sat

    else:
        rule_83(
            solver,
            {
                "values_ndim": values["ndim"],
                "values_shape": values["shape"],
                "axis": axis["value"],
                "direction_valid": direction["valid"]
            },
            neg
        )
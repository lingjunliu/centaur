import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 82: valid axis

rule_82 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["values_ndim"] >= 1,
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
def rule_82_func(arg1, arg2, solver=None, neg=False):
    values = next(iter(arg1.values()))
    axis = next(iter(arg2.values()))

    if not solver:
        if not isinstance(values, np.ndarray):
            return False
        if not isinstance(axis, int):
            return False
        if not (-values.ndim <= axis < values.ndim):
            return False

        solver = Solver()
        values_ndim = Int('values_ndim')
        values_shape = Array('values_shape', IntSort(), IntSort())
        axis_var = Int('axis')

        solver.add(values_ndim == values.ndim)
        solver.add(axis_var == axis)

        for i in range(values.ndim):
            solver.add(Select(values_shape, i) == values.shape[i])

        rule_82(solver, {
            "values_ndim": values_ndim,
            "values_shape": values_shape,
            "axis": axis_var
        })

        return solver.check() == sat

    else:
        rule_82(
            solver,
            {
                "values_ndim": values["ndim"],
                "values_shape": values["shape"],
                "axis": axis["value"]
            },
            neg
        )
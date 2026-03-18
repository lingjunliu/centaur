import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 81: valid direction

rule_81 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["values_ndim"] >= 1,
                v["direction_valid"],
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
            And([
                Implies(i < v["values_ndim"],
                        Select(v["values_shape"], i) > 0)
                for i in range(MAX_N_DIM)
            ])
        )
    )
)
def rule_81_func(arg1, arg2, solver=None, neg=False):
    values = next(iter(arg1.values()))
    direction = next(iter(arg2.values()))

    if not solver:
        if not isinstance(values, np.ndarray):
            return False
        if direction not in ["ASCENDING", "DESCENDING"]:
            return False

        solver = Solver()
        values_ndim = Int('values_ndim')
        values_shape = Array('values_shape', IntSort(), IntSort())

        solver.add(values_ndim == values.ndim)
        for i in range(values.ndim):
            solver.add(Select(values_shape, i) == values.shape[i])

        direction_valid = Bool('direction_valid')
        solver.add(direction_valid == True)

        rule_81(solver, {
            "values_ndim": values_ndim,
            "values_shape": values_shape,
            "direction_valid": direction_valid
        })

        return solver.check() == sat

    else:
        rule_81(
            solver,
            {
                "values_ndim": values["ndim"],
                "values_shape": values["shape"],
                "direction_valid": direction["valid"]
            },
            neg
        )
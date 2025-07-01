import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If v_1 is str has a value in ["tanh", "sum", "mean", "max"] AND tensor v_2 has at least 2 dimensions, then both dimensions' shape at index 0 and index 1 are above 10 (Rule 1453)

rule_1453 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 8), v["arg1_value"] == 7), v["arg1_value"] == 9)), v["arg2_ndim"] >= 2), And(Select(v["arg2_shape"], 0) > 10, Select(v["arg2_shape"], 1) > 10), False)) if n else
          If(And((Or(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 8), v["arg1_value"] == 7), v["arg1_value"] == 9)), v["arg2_ndim"] >= 2), And(Select(v["arg2_shape"], 0) > 10, Select(v["arg2_shape"], 1) > 10), False))
)

def rule_1453_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 1453
        rule_1453(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1453(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)

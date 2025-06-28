import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If a string value is one of "sum", "mean", "max" and the tensor is 3 dimensional, then the product of its dimensions should be greater than 10 (Rule 999)

rule_999 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(v["arg2_value"] == 8, v["arg2_value"] == 7), v["arg2_value"] == 9)), v["arg1_ndim"] == 3), Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) > 10, False)) if n else
          If(And((Or(Or(v["arg2_value"] == 8, v["arg2_value"] == 7), v["arg2_value"] == 9)), v["arg1_ndim"] == 3), Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) > 10, False))
)

def rule_999_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 999
        rule_999(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_999(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If shape of tensor changes dramatically from width to height, then alpha * beta cannot be too large. Some dimension has to be large. (Rule 138)

rule_138 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Select(v["arg1_shape"], 2) - Select(v["arg1_shape"], 3) > 50, Select(v["arg1_shape"], 3) - Select(v["arg1_shape"], 2) > 50)), (Or(Select(v["arg1_shape"], 2) > 5, Select(v["arg1_shape"], 3) > 5))), v["arg2_value"] * v["arg3_value"] < 10, True)) if n else
          If(And((Or(Select(v["arg1_shape"], 2) - Select(v["arg1_shape"], 3) > 50, Select(v["arg1_shape"], 3) - Select(v["arg1_shape"], 2) > 50)), (Or(Select(v["arg1_shape"], 2) > 5, Select(v["arg1_shape"], 3) > 5))), v["arg2_value"] * v["arg3_value"] < 10, True))
)

def rule_138_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 138
        rule_138(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_138(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

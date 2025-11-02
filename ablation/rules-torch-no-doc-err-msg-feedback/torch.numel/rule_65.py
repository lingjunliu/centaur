import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# For a specific boolean parameter, another tensor dimension must follow certain restrictions (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], Select(v["arg2_shape"], v["arg3_value"]) > 10, Select(v["arg2_shape"], v["arg3_value"]) < 10)) if n else
          If(v["arg1_value"], Select(v["arg2_shape"], v["arg3_value"]) > 10, Select(v["arg2_shape"], v["arg3_value"]) < 10))
)

def rule_65_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 65
        rule_65(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

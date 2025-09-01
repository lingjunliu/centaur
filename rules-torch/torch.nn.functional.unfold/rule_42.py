import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Padding must not cause output dimensions to be zero or negative - comprehensive check (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(And((Select(v["arg4_shape"], 2) + 2 * v["arg1_value"] - (Select(v["arg2_values"], 0) - 1) * v["arg3_value"] - 1) >= 0, (Select(v["arg4_shape"], 3) + 2 * v["arg1_value"] - (Select(v["arg2_values"], 1) - 1) * v["arg3_value"] - 1) >= 0)) if n else
          And((Select(v["arg4_shape"], 2) + 2 * v["arg1_value"] - (Select(v["arg2_values"], 0) - 1) * v["arg3_value"] - 1) >= 0, (Select(v["arg4_shape"], 3) + 2 * v["arg1_value"] - (Select(v["arg2_values"], 1) - 1) * v["arg3_value"] - 1) >= 0))
)

def rule_42_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == int(arg3))
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 42
        rule_42(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape']}, neg)

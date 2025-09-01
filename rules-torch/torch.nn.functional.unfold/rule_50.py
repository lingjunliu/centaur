import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Dilation, kernel size and input size must satisfy output size constraints (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(And((Select(v["arg3_shape"], 2) + 2 * v["arg4_value"] - (Select(v["arg2_values"], 0) - 1) * v["arg1_value"] - 1) >= 0, (Select(v["arg3_shape"], 3) + 2 * v["arg4_value"] - (Select(v["arg2_values"], 1) - 1) * v["arg1_value"] - 1) >= 0)) if n else
          And((Select(v["arg3_shape"], 2) + 2 * v["arg4_value"] - (Select(v["arg2_values"], 0) - 1) * v["arg1_value"] - 1) >= 0, (Select(v["arg3_shape"], 3) + 2 * v["arg4_value"] - (Select(v["arg2_values"], 1) - 1) * v["arg1_value"] - 1) >= 0))
)

def rule_50_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 50
        rule_50(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)

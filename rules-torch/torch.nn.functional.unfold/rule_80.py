import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Dilation, kernel size, padding and image size must satisfy output size constraints + stride, and handle zero kernel edge case + stride limit (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] > 0, v["arg5_value"] > 0), (Or((And(And(And(Select(v["arg2_values"], 0) > 0, Select(v["arg2_values"], 1) > 0), (Select(v["arg3_shape"], 2) + 2 * v["arg4_value"] - (Select(v["arg2_values"], 0) - 1) * v["arg1_value"] - 1) / v["arg5_value"] >= 0), (Select(v["arg3_shape"], 3) + 2 * v["arg4_value"] - (Select(v["arg2_values"], 1) - 1) * v["arg1_value"] - 1) / v["arg5_value"] >= 0)), (Or(Select(v["arg2_values"], 0) == 0, Select(v["arg2_values"], 1) == 0)))))) if n else
          And(And(v["arg1_value"] > 0, v["arg5_value"] > 0), (Or((And(And(And(Select(v["arg2_values"], 0) > 0, Select(v["arg2_values"], 1) > 0), (Select(v["arg3_shape"], 2) + 2 * v["arg4_value"] - (Select(v["arg2_values"], 0) - 1) * v["arg1_value"] - 1) / v["arg5_value"] >= 0), (Select(v["arg3_shape"], 3) + 2 * v["arg4_value"] - (Select(v["arg2_values"], 1) - 1) * v["arg1_value"] - 1) / v["arg5_value"] >= 0)), (Or(Select(v["arg2_values"], 0) == 0, Select(v["arg2_values"], 1) == 0))))))
)

def rule_80_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

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
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))

        # Constraints for rule 80
        rule_80(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)

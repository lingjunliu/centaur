import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel_size and padding must result in a valid output dimension with stride and ceil mode (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(If(v["arg5_value"] == False, ((Select(v["arg2_shape"], 2) + 2 * v["arg3_value"] - v["arg1_value"]) / v["arg4_value"]) > 0, ((Select(v["arg2_shape"], 2) + 2 * v["arg3_value"] - v["arg1_value"] + v["arg4_value"] - 1) / v["arg4_value"]) > 0)) if n else
          If(v["arg5_value"] == False, ((Select(v["arg2_shape"], 2) + 2 * v["arg3_value"] - v["arg1_value"]) / v["arg4_value"]) > 0, ((Select(v["arg2_shape"], 2) + 2 * v["arg3_value"] - v["arg1_value"] + v["arg4_value"] - 1) / v["arg4_value"]) > 0))
)

def rule_54_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == arg5)

        # Constraints for rule 54
        rule_54(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)

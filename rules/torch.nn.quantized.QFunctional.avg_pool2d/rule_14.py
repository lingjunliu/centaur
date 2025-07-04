import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If ceil_mode is True, the output shape can potentially be larger than that without ceil_mode. (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And((Select(v["arg5_shape"], 2) >= (Select(v["arg5_shape"], 2) - Select(v["arg2_values"], 0) + 2 * v["arg4_value"]) / Select(v["arg3_values"], 0) + 1), (Select(v["arg5_shape"], 3) >= (Select(v["arg5_shape"], 3) - Select(v["arg2_values"], 1) + 2 * v["arg4_value"]) / Select(v["arg3_values"], 1) + 1)), False)) if n else
          If(v["arg1_value"] == True, And((Select(v["arg5_shape"], 2) >= (Select(v["arg5_shape"], 2) - Select(v["arg2_values"], 0) + 2 * v["arg4_value"]) / Select(v["arg3_values"], 0) + 1), (Select(v["arg5_shape"], 3) >= (Select(v["arg5_shape"], 3) - Select(v["arg2_values"], 1) + 2 * v["arg4_value"]) / Select(v["arg3_values"], 1) + 1)), False))
)

def rule_14_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_value = Int('arg4_value')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_value == int(arg4))
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 14
        rule_14(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_values': arg3_values, 'arg4_value': arg4_value, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values'], 'arg4_value': arg4['value'], 'arg5_shape': arg5['shape']}, neg)

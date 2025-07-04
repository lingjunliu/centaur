import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If ceil_mode is False, the output size cannot be zero. (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == False, And((Select(v["arg2_shape"], 2) - Select(v["arg3_values"], 0) + 2 * v["arg5_value"]) / Select(v["arg4_values"], 0) >= 1, (Select(v["arg2_shape"], 3) - Select(v["arg3_values"], 1) + 2 * v["arg5_value"]) / Select(v["arg4_values"], 1) >= 1), False)) if n else
          If(v["arg1_value"] == False, And((Select(v["arg2_shape"], 2) - Select(v["arg3_values"], 0) + 2 * v["arg5_value"]) / Select(v["arg4_values"], 0) >= 1, (Select(v["arg2_shape"], 3) - Select(v["arg3_values"], 1) + 2 * v["arg5_value"]) / Select(v["arg4_values"], 1) >= 1), False))
)

def rule_23_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_value == int(arg5))

        # Constraints for rule 23
        rule_23(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values, 'arg4_values': arg4_values, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values'], 'arg4_values': arg4['values'], 'arg5_value': arg5['value']}, neg)

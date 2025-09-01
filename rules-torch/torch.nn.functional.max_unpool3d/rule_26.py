import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# input tensor and indices tensor should have compatible shapes given stride and padding (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(And(And(Select(v["arg2_shape"], 2) == (Select(v["arg1_shape"], 2) - 1) * v["arg3_value"] - 2 * v["arg4_value"] + Select(v["arg5_values"], 0), Select(v["arg2_shape"], 3) == (Select(v["arg1_shape"], 3) - 1) * v["arg3_value"] - 2 * v["arg4_value"] + Select(v["arg5_values"], 1)), Select(v["arg2_shape"], 4) == (Select(v["arg1_shape"], 4) - 1) * v["arg3_value"] - 2 * v["arg4_value"] + Select(v["arg5_values"], 2))) if n else
          And(And(Select(v["arg2_shape"], 2) == (Select(v["arg1_shape"], 2) - 1) * v["arg3_value"] - 2 * v["arg4_value"] + Select(v["arg5_values"], 0), Select(v["arg2_shape"], 3) == (Select(v["arg1_shape"], 3) - 1) * v["arg3_value"] - 2 * v["arg4_value"] + Select(v["arg5_values"], 1)), Select(v["arg2_shape"], 4) == (Select(v["arg1_shape"], 4) - 1) * v["arg3_value"] - 2 * v["arg4_value"] + Select(v["arg5_values"], 2)))
)

def rule_26_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not (isinstance(arg5, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_values = Array('arg5_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])

        # Constraints for rule 26
        rule_26(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_values': arg5_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_values': arg5['values']}, neg)

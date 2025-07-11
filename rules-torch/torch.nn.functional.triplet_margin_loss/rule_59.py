import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Margin must be positive if P is more than 0 and tensors have the same shape (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg2_value"] > 0, Select(v["arg3_shape"], 0) == Select(v["arg4_shape"], 0)), Select(v["arg4_shape"], 0) == Select(v["arg5_shape"], 0)), v["arg1_value"] > 0, False)) if n else
          If(And(And(v["arg2_value"] > 0, Select(v["arg3_shape"], 0) == Select(v["arg4_shape"], 0)), Select(v["arg4_shape"], 0) == Select(v["arg5_shape"], 0)), v["arg1_value"] > 0, False))
)

def rule_59_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 59
        rule_59(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg4_shape': arg4_shape, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg4_shape': arg4['shape'], 'arg5_shape': arg5['shape']}, neg)

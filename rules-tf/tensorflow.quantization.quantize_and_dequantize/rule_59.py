import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if axis is specified, input_min and input_max should have same shape (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 0, And(Select(v["arg2_shape"], 0) == Select(v["arg3_shape"], 0), Select(v["arg2_shape"], 0) == Select(v["arg4_shape"], v["arg1_value"])), False)) if n else
          If(v["arg1_value"] > 0, And(Select(v["arg2_shape"], 0) == Select(v["arg3_shape"], 0), Select(v["arg2_shape"], 0) == Select(v["arg4_shape"], v["arg1_value"])), False))
)

def rule_59_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 59
        rule_59(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_shape': arg4['shape']}, neg)

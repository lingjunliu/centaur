import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# When no weight is provided the max value of the input tensor needs to be smaller than the value specified in the minLength. (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg3_shape"], 0) == 0, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_range"], 1) < v["arg2_value"], True)) if n else
          If(And(Select(v["arg3_shape"], 0) == 0, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_range"], 1) < v["arg2_value"], True))
)

def rule_67_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 67
        rule_67(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)

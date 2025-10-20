import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Indices must be within the valid range for data if segment_ids is also within valid range for its type and non-empty (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg2_shape"], 0) > 0, v["arg2_dtype"] == 3), Select(v["arg1_range"], 1) < Select(v["arg1_shape"], 0), If(And(Select(v["arg2_shape"], 0) > 0, v["arg2_dtype"] == 4), Select(v["arg1_range"], 1) < Select(v["arg1_shape"], 0), True))) if n else
          If(And(Select(v["arg2_shape"], 0) > 0, v["arg2_dtype"] == 3), Select(v["arg1_range"], 1) < Select(v["arg1_shape"], 0), If(And(Select(v["arg2_shape"], 0) > 0, v["arg2_dtype"] == 4), Select(v["arg1_range"], 1) < Select(v["arg1_shape"], 0), True)))
)

def rule_67_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 67
        rule_67(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)

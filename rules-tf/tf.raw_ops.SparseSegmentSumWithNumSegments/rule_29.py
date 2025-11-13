import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if num_segments is 0, and segment_ids is also empty, the operation is valid, otherwise, all other values must be zero (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg4_range"], 1) == 0, Select(v["arg3_shape"], 0) == 0), True, If(And(Select(v["arg4_range"], 1) == 0, Select(v["arg3_shape"], 0) > 0), And(And(Select(v["arg2_range"], 1) == 0, Select(v["arg3_range"], 1) == 0), Select(v["arg1_shape"], 0) == 0), True))) if n else
          If(And(Select(v["arg4_range"], 1) == 0, Select(v["arg3_shape"], 0) == 0), True, If(And(Select(v["arg4_range"], 1) == 0, Select(v["arg3_shape"], 0) > 0), And(And(Select(v["arg2_range"], 1) == 0, Select(v["arg3_range"], 1) == 0), Select(v["arg1_shape"], 0) == 0), True)))
)

def rule_29_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_range = Array('arg4_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 29
        rule_29(solver, {'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg3_shape': arg3_shape, 'arg3_range': arg3_range, 'arg4_range': arg4_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg3_shape': arg3['shape'], 'arg3_range': arg3['range'], 'arg4_range': arg4['range']}, neg)

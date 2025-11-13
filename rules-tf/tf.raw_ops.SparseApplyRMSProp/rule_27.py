import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The maximum value in indices must be less than the size of the first dimension of var, ms, mom, and grad. (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(And(And(And(Select(v["arg2_range"], 1) < Select(v["arg1_shape"], 0), Select(v["arg2_range"], 1) < Select(v["arg3_shape"], 0)), Select(v["arg2_range"], 1) < Select(v["arg4_shape"], 0)), Select(v["arg2_range"], 1) < Select(v["arg5_shape"], 0))) if n else
          And(And(And(Select(v["arg2_range"], 1) < Select(v["arg1_shape"], 0), Select(v["arg2_range"], 1) < Select(v["arg3_shape"], 0)), Select(v["arg2_range"], 1) < Select(v["arg4_shape"], 0)), Select(v["arg2_range"], 1) < Select(v["arg5_shape"], 0)))
)

def rule_27_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 27
        rule_27(solver, {'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg3_shape': arg3_shape, 'arg4_shape': arg4_shape, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg3_shape': arg3['shape'], 'arg4_shape': arg4['shape'], 'arg5_shape': arg5['shape']}, neg)

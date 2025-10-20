import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The maximum value of the segment ids must be smaller than or equal to number of rows in data pointed to by indices (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_shape"], 0) > 0, Select(v["arg3_range"], 1) <= Select(v["arg1_shape"], 0), True)) if n else
          If(Select(v["arg2_shape"], 0) > 0, Select(v["arg3_range"], 1) <= Select(v["arg1_shape"], 0), True))
)

def rule_39_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 39
        rule_39(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_range': arg3['range']}, neg)

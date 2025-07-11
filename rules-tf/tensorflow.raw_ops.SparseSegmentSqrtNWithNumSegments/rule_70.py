import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if indices is not empty, then num_segments must be greater than the maximum segment_id (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_shape"], 0) > 0, Select(v["arg1_range"], 1) > Select(v["arg3_range"], 1), False)) if n else
          If(Select(v["arg2_shape"], 0) > 0, Select(v["arg1_range"], 1) > Select(v["arg3_range"], 1), False))
)

def rule_70_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 70
        rule_70(solver, {'arg1_range': arg1_range, 'arg2_shape': arg2_shape, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_range': arg1['range'], 'arg2_shape': arg2['shape'], 'arg3_range': arg3['range']}, neg)

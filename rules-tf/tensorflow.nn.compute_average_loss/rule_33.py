import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If sample_weight and global_batch_size are given and shape of sample_weight's first dimension is greater than 0 and global_batch_size is greater than 0, the min of sample_weight must be non-negative (equivalent to all values being non-negative (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_shape"], 0) > 0, v["arg2_value"] > 0), Select(v["arg1_range"], 0) >= 0, False)) if n else
          If(And(Select(v["arg1_shape"], 0) > 0, v["arg2_value"] > 0), Select(v["arg1_range"], 0) >= 0, False))
)

def rule_33_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 33
        rule_33(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)

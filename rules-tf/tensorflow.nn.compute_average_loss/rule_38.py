import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if sample_weight is None, global_batch_size must be positive or per_example_loss must be 0. (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_shape"], 0) == 0, (Or(v["arg3_value"] > 0, (And(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 1) == 0)))), False)) if n else
          If(Select(v["arg2_shape"], 0) == 0, (Or(v["arg3_value"] > 0, (And(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 1) == 0)))), False))
)

def rule_38_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 38
        rule_38(solver, {'arg1_range': arg1_range, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_range': arg1['range'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If number of segments is passed and segment_ids is non-empty, the maximum segment id should be less than number segments -1  (Rule 124)

rule_124 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_shape"], 0) > 0, v["arg1_value"] > Select(v["arg2_range"], 1), True)) if n else
          If(Select(v["arg2_shape"], 0) > 0, v["arg1_value"] > Select(v["arg2_range"], 1), True))
)

def rule_124_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 124
        rule_124(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_124(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_shape': arg2['shape']}, neg)

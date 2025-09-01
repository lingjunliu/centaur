import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the maximum value of segment_ids is greater than zero, then the first dimension of the data tensor must be greater than zero. (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_range"], 1) > 0, Select(v["arg1_shape"], 0) > 0, True)) if n else
          If(Select(v["arg2_range"], 1) > 0, Select(v["arg1_shape"], 0) > 0, True))
)

def rule_56_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 56
        rule_56(solver, {'arg1_shape': arg1_shape, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_shape': arg1['shape'], 'arg2_range': arg2['range']}, neg)

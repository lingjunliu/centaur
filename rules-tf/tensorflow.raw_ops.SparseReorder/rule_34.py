import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the maximum value in the input_indices is greater than 100 and if the first dimension of input_shape is more than 0, then the shape of the input shape in the last dimension should be more than 100 (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_range"], 1) > 100, Select(v["arg2_shape"], 0) > 0), Select(v["arg2_shape"], Select(v["arg2_shape"], 0) - 1) > 100, True)) if n else
          If(And(Select(v["arg1_range"], 1) > 100, Select(v["arg2_shape"], 0) > 0), Select(v["arg2_shape"], Select(v["arg2_shape"], 0) - 1) > 100, True))
)

def rule_34_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 34
        rule_34(solver, {'arg1_range': arg1_range, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_range': arg1['range'], 'arg2_shape': arg2['shape']}, neg)

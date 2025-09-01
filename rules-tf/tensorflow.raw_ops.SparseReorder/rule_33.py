import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the maximum value in the input_indices is greater than 100, then the corresponding dimension size in input_shape should be at least 100. (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 1) > 100, Select(v["arg2_shape"], Select(v["arg1_shape"], 1) - 1) >= 100, True)) if n else
          If(Select(v["arg1_range"], 1) > 100, Select(v["arg2_shape"], Select(v["arg1_shape"], 1) - 1) >= 100, True))
)

def rule_33_func(arg1, arg2, solver=None, neg=False):
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

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 33
        rule_33(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg2_shape': arg2['shape']}, neg)

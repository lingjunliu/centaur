import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the first dimension of params is 1, the maximum value of indices must be 0 if validate_indices is true (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(If((And(Select(v["arg1_shape"], 0) == 1, v["arg3_value"] == True)), Select(v["arg2_range"], 1) == 0, False)) if n else
          If((And(Select(v["arg1_shape"], 0) == 1, v["arg3_value"] == True)), Select(v["arg2_range"], 1) == 0, False))
)

def rule_78_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 78
        rule_78(solver, {'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value']}, neg)

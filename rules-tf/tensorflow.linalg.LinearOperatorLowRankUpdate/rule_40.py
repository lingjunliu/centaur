import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If L and D are positive definite and U equals V, solves and determinants can be attempted (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_range"], 0) > 0, Select(v["arg2_range"], 0) > 0), If(Select(v["arg3_shape"], 0) == Select(v["arg3_shape"], 1), True, False), False)) if n else
          If(And(Select(v["arg1_range"], 0) > 0, Select(v["arg2_range"], 0) > 0), If(Select(v["arg3_shape"], 0) == Select(v["arg3_shape"], 1), True, False), False))
)

def rule_40_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 40
        rule_40(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg3_shape': arg3['shape']}, neg)

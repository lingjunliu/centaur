import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The size of the max indices must be less or equal to the unpool size if unpoolsize given (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_length"] > 0, Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3), False)) if n else
          If(v["arg3_length"] > 0, Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3), False))
)

def rule_17_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_length = Int('arg3_length')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 17
        rule_17(solver, {'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg3_length': arg3['length']}, neg)

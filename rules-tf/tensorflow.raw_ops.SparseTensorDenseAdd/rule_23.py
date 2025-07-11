import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# max value of a_indices must be smaller than shape of b at corresponding dimension (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(And([Implies(j < (Select(v["arg2_shape"], 0) - 1 + 1), Select(v["arg1_range"], 1) < Select(v["arg3_shape"], j)) for j in range(6)])) if n else
          And([Implies(j < (Select(v["arg2_shape"], 0) - 1 + 1), Select(v["arg1_range"], 1) < Select(v["arg3_shape"], j)) for j in range(6)]))
)

def rule_23_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 23
        rule_23(solver, {'arg1_range': arg1_range, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_range': arg1['range'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape']}, neg)

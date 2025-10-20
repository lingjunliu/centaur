import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if min of the first tensor is less than 0 and dimension of second tensor is greater than 5, shape must be different from 0 (Rule 114)

rule_114 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_range"], 0) < 0, v["arg2_ndim"] > 5), Select(v["arg1_shape"], 0) != 0, True)) if n else
          If(And(Select(v["arg1_range"], 0) < 0, v["arg2_ndim"] > 5), Select(v["arg1_shape"], 0) != 0, True))
)

def rule_114_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 114
        rule_114(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_114(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim']}, neg)

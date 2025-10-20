import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The dimension of first tensor times the maximum value of first tensor must be greater than the shape of the second tensor (Rule 122)

rule_122 = lambda s, v, n=False: (
    s.add(Not(v["arg1_ndim"] * Select(v["arg1_range"], 1) > Select(v["arg2_shape"], 0)) if n else
          v["arg1_ndim"] * Select(v["arg1_range"], 1) > Select(v["arg2_shape"], 0))
)

def rule_122_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 122
        rule_122(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_122(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The paddings tensor shape must be [Dn, 2] where Dn is the rank of input tensor, and paddings must be non-negative. (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(And(And(Select(v["arg2_shape"], 0) == v["arg1_ndim"], Select(v["arg2_shape"], 1) == 2), Select(v["arg2_range"], 0) >= 0)) if n else
          And(And(Select(v["arg2_shape"], 0) == v["arg1_ndim"], Select(v["arg2_shape"], 1) == 2), Select(v["arg2_range"], 0) >= 0))
)

def rule_63_func(arg1, arg2, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 63
        rule_63(solver, {'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range'], 'arg2_shape': arg2['shape']}, neg)

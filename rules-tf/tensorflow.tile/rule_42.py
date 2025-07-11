import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If multiples is all 1s, then the output will have same data as input (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(If(And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 1) for i in range(6)]), Select(v["arg1_range"], 0) == Select(v["arg1_range"], 0), False)) if n else
          If(And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 1) for i in range(6)]), Select(v["arg1_range"], 0) == Select(v["arg1_range"], 0), False))
)

def rule_42_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 42
        rule_42(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape']}, neg)

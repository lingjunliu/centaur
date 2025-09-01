import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If a is a scalar, all elements of x must be non-negative, and if x is a scalar, all elements of a must be positive, otherwise min(a (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, Select(v["arg2_range"], 0) >= 0, If(v["arg2_ndim"] == 0, Select(v["arg1_range"], 0) > 0, And(Select(v["arg1_range"], 0) > 0, Select(v["arg2_range"], 0) >= 0)))) if n else
          If(v["arg1_ndim"] == 0, Select(v["arg2_range"], 0) >= 0, If(v["arg2_ndim"] == 0, Select(v["arg1_range"], 0) > 0, And(Select(v["arg1_range"], 0) > 0, Select(v["arg2_range"], 0) >= 0))))
)

def rule_68_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 68
        rule_68(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_ndim': arg2_ndim, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_ndim': arg2['ndim'], 'arg2_range': arg2['range']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If y is a scalar, then x's minimum must be greater than or equal to zero (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 0, Select(v["arg1_range"], 0) >= 0, False)) if n else
          If(v["arg2_ndim"] == 0, Select(v["arg1_range"], 0) >= 0, False))
)

def rule_31_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 31
        rule_31(solver, {'arg1_range': arg1_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_range': arg1['range'], 'arg2_ndim': arg2['ndim']}, neg)

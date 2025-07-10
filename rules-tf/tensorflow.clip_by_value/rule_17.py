import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If clip_value_max is a scalar tensor, then clip_value_min should be less than or equal to the max value of t. (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_ndim"] == 0, Select(v["arg2_range"], 0) <= Select(v["arg1_range"], 1), False)) if n else
          If(v["arg3_ndim"] == 0, Select(v["arg2_range"], 0) <= Select(v["arg1_range"], 1), False))
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
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 17
        rule_17(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg3_ndim': arg3['ndim']}, neg)

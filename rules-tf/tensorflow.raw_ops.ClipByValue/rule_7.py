import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If clip_value_min is scalar, its value should be less than or equal to the maximum of clip_value_max (Rule 7)

rule_7 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 0, Select(v["arg2_range"], 0) <= Select(v["arg3_range"], 1), True)) if n else
          If(v["arg2_ndim"] == 0, Select(v["arg2_range"], 0) <= Select(v["arg3_range"], 1), True))
)

def rule_7_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 7
        rule_7(solver, {'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_7(solver, {'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg3_range': arg3['range']}, neg)

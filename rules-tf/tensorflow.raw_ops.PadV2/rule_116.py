import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Constant_values must be a scalar, and its data type must match that of the Input tensor, and Input Data Type should be in valid range. Constant Values range should be limited (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg2_ndim"] == 0, v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] >= 0), v["arg1_dtype"] <= 12), Select(v["arg2_range"], 0) >= -2048), Select(v["arg2_range"], 1) <= 2048)) if n else
          And(And(And(And(And(v["arg2_ndim"] == 0, v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] >= 0), v["arg1_dtype"] <= 12), Select(v["arg2_range"], 0) >= -2048), Select(v["arg2_range"], 1) <= 2048))
)

def rule_116_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 116
        rule_116(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)

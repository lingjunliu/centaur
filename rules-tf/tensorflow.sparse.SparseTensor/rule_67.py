import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Combined rule about dimensions, type and non-negativity of dense_shape and indices (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_dtype"] == 4, Select(v["arg2_range"], 0) >= 0), v["arg2_ndim"] > 0)) if n else
          And(And(v["arg1_dtype"] == 4, Select(v["arg2_range"], 0) >= 0), v["arg2_ndim"] > 0))
)

def rule_67_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 67
        rule_67(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)

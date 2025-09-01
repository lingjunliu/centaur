import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Targets must have int32 or int64 datatype and their values must fall in their corresponding range if prediction has 2 dimensions. (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 2, Or((And(And(v["arg2_dtype"] == 3, Select(v["arg2_range"], 0) >= -2147483648), Select(v["arg2_range"], 1) <= 2147483647)), (And(And(v["arg2_dtype"] == 4, Select(v["arg2_range"], 0) >= -9223372036854775808), Select(v["arg2_range"], 1) <= 9223372036854775807))), True)) if n else
          If(v["arg1_ndim"] == 2, Or((And(And(v["arg2_dtype"] == 3, Select(v["arg2_range"], 0) >= -2147483648), Select(v["arg2_range"], 1) <= 2147483647)), (And(And(v["arg2_dtype"] == 4, Select(v["arg2_range"], 0) >= -9223372036854775808), Select(v["arg2_range"], 1) <= 9223372036854775807))), True))
)

def rule_51_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 51
        rule_51(solver, {'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype']}, neg)

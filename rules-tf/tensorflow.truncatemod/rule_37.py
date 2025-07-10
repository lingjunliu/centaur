import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If x's dtype is int32, y's values must be between -2147483648 and 2147483647 (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 2, And(Select(v["arg2_range"], 0) >= -2147483648, Select(v["arg2_range"], 1) <= 2147483647), False)) if n else
          If(v["arg1_dtype"] == 2, And(Select(v["arg2_range"], 0) >= -2147483648, Select(v["arg2_range"], 1) <= 2147483647), False))
)

def rule_37_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 37
        rule_37(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range']}, neg)

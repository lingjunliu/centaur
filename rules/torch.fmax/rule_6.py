import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if other tensor is integer type, the input tensor's minimum value cannot be less than the smallest possible integer for that dtype (Rule 6)

rule_6 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 1, Select(v["arg1_range"], 0) >= -128, If(v["arg2_dtype"] == 2, Select(v["arg1_range"], 0) >= -32768, If(v["arg2_dtype"] == 3, Select(v["arg1_range"], 0) >= -2147483648, False)))) if n else
          If(v["arg2_dtype"] == 1, Select(v["arg1_range"], 0) >= -128, If(v["arg2_dtype"] == 2, Select(v["arg1_range"], 0) >= -32768, If(v["arg2_dtype"] == 3, Select(v["arg1_range"], 0) >= -2147483648, False))))
)

def rule_6_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 6
        rule_6(solver, {'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(solver, {'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype']}, neg)

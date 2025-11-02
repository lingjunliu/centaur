import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the base is a tensor with complex numbers and the exponent is a real tensor, make sure that the absolute value of exponent does not exceed a threshold value to prevent overflows when converted to complex type during power computation. (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), If(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), Select(v["arg2_range"], 1) < 1000, True), True)) if n else
          If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), If(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), Select(v["arg2_range"], 1) < 1000, True), True))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 46
        rule_46(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype']}, neg)

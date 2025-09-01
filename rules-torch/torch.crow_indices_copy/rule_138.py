import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Indices are valid and smaller than a number given by limits based on dtype of tensor (Rule 138)

rule_138 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, Select(v["arg1_range"], 1) < 127, If(v["arg1_dtype"] == 2, Select(v["arg1_range"], 1) < 32767, If(v["arg1_dtype"] == 3, Select(v["arg1_range"], 1) < 2147483647, If(v["arg1_dtype"] == 4, Select(v["arg1_range"], 1) < 9223372036854775807, If(v["arg1_dtype"] == 5, Select(v["arg1_range"], 1) < 255, True)))))) if n else
          If(v["arg1_dtype"] == 1, Select(v["arg1_range"], 1) < 127, If(v["arg1_dtype"] == 2, Select(v["arg1_range"], 1) < 32767, If(v["arg1_dtype"] == 3, Select(v["arg1_range"], 1) < 2147483647, If(v["arg1_dtype"] == 4, Select(v["arg1_range"], 1) < 9223372036854775807, If(v["arg1_dtype"] == 5, Select(v["arg1_range"], 1) < 255, True))))))
)

def rule_138_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 138
        rule_138(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_138(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']}, neg)

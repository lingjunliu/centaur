import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# For Tensor, Tensor: if the output dtype is Byte, the result values after power must be representable as Byte (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_dtype"] == 1, And(And(And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) < 256), Select(v["arg2_range"], 0) >= 0), Select(v["arg2_range"], 1) < 8), True)) if n else
          If(v["arg3_dtype"] == 1, And(And(And(Select(v["arg1_range"], 0) >= 0, Select(v["arg1_range"], 1) < 256), Select(v["arg2_range"], 0) >= 0), Select(v["arg2_range"], 1) < 8), True))
)

def rule_62_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 62
        rule_62(solver, {'arg1_range': arg1_range, 'arg2_range': arg2_range, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_range': arg1['range'], 'arg2_range': arg2['range'], 'arg3_dtype': arg3['dtype']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dims contains extremely large values, value tensor can not have dtype float16 and float32 as it may not have the capacity to store them (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 1) > 1000000, And(v["arg2_dtype"] != 6, v["arg2_dtype"] != 7), False)) if n else
          If(Select(v["arg1_range"], 1) > 1000000, And(v["arg2_dtype"] != 6, v["arg2_dtype"] != 7), False))
)

def rule_69_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 69
        rule_69(solver, {'arg1_range': arg1_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_range': arg1['range'], 'arg2_dtype': arg2['dtype']}, neg)

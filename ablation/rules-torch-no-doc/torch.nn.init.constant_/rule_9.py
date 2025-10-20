import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# value should be a number that can be safely cast to float16 without overflow if the tensor's dtype is float16 (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And(And(v["arg2_value"] >= -65504.0, v["arg2_value"] <= 65504.0), (And(Select(v["arg1_range"], 0) >= -65504.0, Select(v["arg1_range"], 1) <= 65504.0))), True)) if n else
          If(v["arg1_dtype"] == 6, And(And(v["arg2_value"] >= -65504.0, v["arg2_value"] <= 65504.0), (And(Select(v["arg1_range"], 0) >= -65504.0, Select(v["arg1_range"], 1) <= 65504.0))), True))
)

def rule_9_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 9
        rule_9(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)

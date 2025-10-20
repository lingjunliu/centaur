import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if the tensor's dtype is an integer type, then the value must be an integer (Rule 82)

rule_82 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5)), Or([And(i < (9223372036854775807 + 1), i == v["arg2_value"]) for i in range(6)]), True)) if n else
          If((And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5)), Or([And(i < (9223372036854775807 + 1), i == v["arg2_value"]) for i in range(6)]), True))
)

def rule_82_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 82
        rule_82(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_82(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)

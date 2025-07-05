import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Example requires tensor to have a dtype between 6 and 8 and v2 should be a bool set to true and third variable is int between 1 and 10 (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8), v["arg2_value"] == True), v["arg3_value"] > 1), v["arg3_value"] < 10)) if n else
          And(And(And(And(v["arg1_dtype"] >= 6, v["arg1_dtype"] <= 8), v["arg2_value"] == True), v["arg3_value"] > 1), v["arg3_value"] < 10))
)

def rule_116_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 116
        rule_116(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

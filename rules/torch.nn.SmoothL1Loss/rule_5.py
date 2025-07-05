import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Ensure beta is not zero when reduction is mean/sum to avoid potential division by zero errors during gradient calculation when using L2 portion (Rule 5)

rule_5 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg1_value"] > 0, False)) if n else
          If(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg1_value"] > 0, False))
)

def rule_5_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 5
        rule_5(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

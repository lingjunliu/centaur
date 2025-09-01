import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If both are numbers, ensure they are within reasonable bounds to avoid overflow/underflow during type promotion. (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] < 1000000000.0, v["arg1_value"] > -1000000000.0), v["arg2_value"] < 1000000000.0), v["arg2_value"] > -1000000000.0)) if n else
          And(And(And(v["arg1_value"] < 1000000000.0, v["arg1_value"] > -1000000000.0), v["arg2_value"] < 1000000000.0), v["arg2_value"] > -1000000000.0))
)

def rule_14_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 14
        rule_14(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

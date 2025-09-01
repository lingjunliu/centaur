import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If axis is not int, type error. (Rule 112)

rule_112 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_value"] == (v["arg1_value"] / 1) * 1), (v["arg2_value"] == (v["arg2_value"] / 1) * 1))) if n else
          And((v["arg1_value"] == (v["arg1_value"] / 1) * 1), (v["arg2_value"] == (v["arg2_value"] / 1) * 1)))
)

def rule_112_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 112
        rule_112(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_112(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

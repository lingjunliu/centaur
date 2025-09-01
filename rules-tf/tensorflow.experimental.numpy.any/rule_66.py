import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Axis must be an integer or a string with value equal to "none" (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg2_value"] == 6), (Or([And(i < (31 + 1), v["arg2_value"] == i) for i in range(6)])))) if n else
          Or((v["arg2_value"] == 6), (Or([And(i < (31 + 1), v["arg2_value"] == i) for i in range(6)]))))
)

def rule_66_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 66
        rule_66(solver, {'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg2_value': arg2['value']}, neg)

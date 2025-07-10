import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# validate_indices is a bool or it is not provided (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg2_value"] == True, v["arg2_value"] == False), v["arg2_value"] == 0)) if n else
          Or(Or(v["arg2_value"] == True, v["arg2_value"] == False), v["arg2_value"] == 0))
)

def rule_45_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, bool) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 45
        rule_45(solver, {'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg2_value': arg2['value']}, neg)

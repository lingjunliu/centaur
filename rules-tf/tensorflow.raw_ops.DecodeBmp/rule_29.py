import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if channels is an int, it should be either 0, 3, or 4; if it's a string, it should be "none" (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 6, True, (Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 3), v["arg1_value"] == 4)))) if n else
          If(v["arg1_value"] == 6, True, (Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 3), v["arg1_value"] == 4))))
)

def rule_29_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 29
        rule_29(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_value': arg1['value']}, neg)

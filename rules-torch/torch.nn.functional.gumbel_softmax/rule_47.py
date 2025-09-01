import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Hard should be a boolean to ensure proper operation, allow int for convenience (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] == 0), v["arg1_value"] == 1)) if n else
          Or(Or(Or(v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] == 0), v["arg1_value"] == 1))
)

def rule_47_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 47
        rule_47(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_value': arg1['value']}, neg)

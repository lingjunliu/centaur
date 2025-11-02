import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The value should be in range of 0 to 2 (Rule 137)

rule_137 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] >= 0, v["arg1_value"] <= 2)) if n else
          And(v["arg1_value"] >= 0, v["arg1_value"] <= 2))
)

def rule_137_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 137
        rule_137(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_137(solver, {'arg1_value': arg1['value']}, neg)

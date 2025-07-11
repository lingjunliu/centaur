import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# The dimension must be an integer or None (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] == (v["arg1_value"] + 0), v["arg1_value"] == 6)) if n else
          Or(v["arg1_value"] == (v["arg1_value"] + 0), v["arg1_value"] == 6))
)

def rule_59_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 59
        rule_59(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_value': arg1['value']}, neg)

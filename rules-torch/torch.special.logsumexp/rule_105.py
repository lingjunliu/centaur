import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The dimension must be an integer or None (Rule 105)

rule_105 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] > (v["arg1_value"] - 1), v["arg1_value"] < (v["arg1_value"] + 1)), True, v["arg1_value"] == 6)) if n else
          If(And(v["arg1_value"] > (v["arg1_value"] - 1), v["arg1_value"] < (v["arg1_value"] + 1)), True, v["arg1_value"] == 6))
)

def rule_105_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 105
        rule_105(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_105(solver, {'arg1_value': arg1['value']}, neg)

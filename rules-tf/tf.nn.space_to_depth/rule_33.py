import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# block_size should be a non-negative integer and at least 2 (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] >= 2, (v["arg1_value"] - (v["arg1_value"] % 1)) == v["arg1_value"]), v["arg1_value"] == (v["arg1_value"] * 1.0))) if n else
          And(And(v["arg1_value"] >= 2, (v["arg1_value"] - (v["arg1_value"] % 1)) == v["arg1_value"]), v["arg1_value"] == (v["arg1_value"] * 1.0)))
)

def rule_33_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 33
        rule_33(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_value': arg1['value']}, neg)

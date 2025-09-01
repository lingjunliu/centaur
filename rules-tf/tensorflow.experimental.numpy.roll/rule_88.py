import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The shift should not be too close to the maximum possible value of integer, it may result in overflow, shift has to be a valid integer and with limits of size  (Rule 88)

rule_88 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] < 100000000, v["arg1_value"] > -100000000), v["arg1_value"] == (v["arg1_value"] + 0 - v["arg1_value"]))) if n else
          And(And(v["arg1_value"] < 100000000, v["arg1_value"] > -100000000), v["arg1_value"] == (v["arg1_value"] + 0 - v["arg1_value"])))
)

def rule_88_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 88
        rule_88(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_88(solver, {'arg1_value': arg1['value']}, neg)

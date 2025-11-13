import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Axis can be only 'none' or valid integer (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg1_value"] == 6), (And((v["arg1_value"] > (0 - 1000)), (v["arg1_value"] < 1000))))) if n else
          Or((v["arg1_value"] == 6), (And((v["arg1_value"] > (0 - 1000)), (v["arg1_value"] < 1000)))))
)

def rule_100_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 100
        rule_100(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_value': arg1['value']}, neg)

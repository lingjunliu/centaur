import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Scalar spacing can not be negative infinity (Rule 81)

rule_81 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] > -100000000000000) if n else
          v["arg1_value"] > -100000000000000)
)

def rule_81_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 81
        rule_81(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_81(solver, {'arg1_value': arg1['value']}, neg)

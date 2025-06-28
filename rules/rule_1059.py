import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# When the shape is specified in AdaptiveAvgPool2D, the output size must be greater or equal than 1 (Rule 1059)

rule_1059 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] >= 1) if n else
          v["arg1_value"] >= 1)
)

def rule_1059_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 1059
        rule_1059(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1059(solver, {'arg1_value': arg1['value']}, neg)

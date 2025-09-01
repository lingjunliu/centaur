import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Dimension must be an integer value (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(Or([And(x < (1 + 1), (v["arg1_value"] * x) % x == 0) for x in range(6)])) if n else
          Or([And(x < (1 + 1), (v["arg1_value"] * x) % x == 0) for x in range(6)]))
)

def rule_32_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 32
        rule_32(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_value': arg1['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If no h_0 is provided, then hidden_size cannot exceed int32 limit/4 to ensure weight/bias tensor sizes are within bounds (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] < 536870912) if n else
          v["arg1_value"] < 536870912)
)

def rule_58_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 58
        rule_58(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_value': arg1['value']}, neg)

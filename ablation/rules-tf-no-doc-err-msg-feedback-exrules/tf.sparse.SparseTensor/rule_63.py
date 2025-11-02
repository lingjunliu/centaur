import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The 'axis' parameter of sparse_fill_empty_rows must be 0 or 1. (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg1_value"] == 0), (v["arg1_value"] == 1))) if n else
          Or((v["arg1_value"] == 0), (v["arg1_value"] == 1)))
)

def rule_63_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 63
        rule_63(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_value': arg1['value']}, neg)

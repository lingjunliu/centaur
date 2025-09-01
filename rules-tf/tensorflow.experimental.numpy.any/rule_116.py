import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The integer axis cannot be too large or too small in absolute value (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(And((v["arg2_value"] > (0 - 32 - 5)), (v["arg2_value"] < (32 + 5)))) if n else
          And((v["arg2_value"] > (0 - 32 - 5)), (v["arg2_value"] < (32 + 5))))
)

def rule_116_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 116
        rule_116(solver, {'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg2_value': arg2['value']}, neg)

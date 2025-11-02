import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Make sure that both numbers can be cast to boolean (Rule 128)

rule_128 = lambda s, v, n=False: (
    s.add(Not(And(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), Or(v["arg2_value"] == 0, v["arg2_value"] == 1))) if n else
          And(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), Or(v["arg2_value"] == 0, v["arg2_value"] == 1)))
)

def rule_128_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 128
        rule_128(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_128(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the assigned value is positive, exclusive access should be set to true; otherwise, to false. (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 0, v["arg2_value"] == True, v["arg2_value"] == False)) if n else
          If(v["arg1_value"] > 0, v["arg2_value"] == True, v["arg2_value"] == False))
)

def rule_64_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg2_value == arg2)

        # Constraints for rule 64
        rule_64(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

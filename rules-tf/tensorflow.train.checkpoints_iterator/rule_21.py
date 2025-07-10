import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If both timeout and min_interval_secs are zero, the program will hang. (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, v["arg2_value"] > 0, False)) if n else
          If(v["arg1_value"] == 0, v["arg2_value"] > 0, False))
)

def rule_21_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg2_value == arg2)

        # Constraints for rule 21
        rule_21(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

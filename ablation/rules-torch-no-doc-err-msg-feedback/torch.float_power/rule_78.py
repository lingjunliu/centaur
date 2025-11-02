import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If base is 0 and exponent is negative, the result is +/- infinity depending on whether base is positive or negative zero (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, If(v["arg2_value"] < 0, True, True), True)) if n else
          If(v["arg1_value"] == 0, If(v["arg2_value"] < 0, True, True), True))
)

def rule_78_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not (isinstance(arg2, (float, np.floating)) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 78
        rule_78(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

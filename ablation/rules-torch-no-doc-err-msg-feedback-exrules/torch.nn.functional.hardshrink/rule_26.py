import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Lambd should be less than or equal to the maximum possible value that the data type can represent. This prevents overflow. (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 7, v["arg1_value"] <= 3.4028235e+38, If(v["arg2_dtype"] == 8, v["arg1_value"] <= 1.7976931348623157e+308, True))) if n else
          If(v["arg2_dtype"] == 7, v["arg1_value"] <= 3.4028235e+38, If(v["arg2_dtype"] == 8, v["arg1_value"] <= 1.7976931348623157e+308, True)))
)

def rule_26_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 26
        rule_26(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)

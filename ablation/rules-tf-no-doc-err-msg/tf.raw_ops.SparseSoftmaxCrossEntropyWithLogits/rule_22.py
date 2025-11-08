import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The number of classes should be less than the maximum value for the indices dtype (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 1, v["arg1_value"] < 128, If(v["arg2_dtype"] == 2, v["arg1_value"] < 32768, If(v["arg2_dtype"] == 3, v["arg1_value"] < 2147483648, If(v["arg2_dtype"] == 4, v["arg1_value"] < 9223372036854775808, If(v["arg2_dtype"] == 5, v["arg1_value"] < 256, True)))))) if n else
          If(v["arg2_dtype"] == 1, v["arg1_value"] < 128, If(v["arg2_dtype"] == 2, v["arg1_value"] < 32768, If(v["arg2_dtype"] == 3, v["arg1_value"] < 2147483648, If(v["arg2_dtype"] == 4, v["arg1_value"] < 9223372036854775808, If(v["arg2_dtype"] == 5, v["arg1_value"] < 256, True))))))
)

def rule_22_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 22
        rule_22(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)

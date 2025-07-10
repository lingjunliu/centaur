import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If dtype is floating point and maxval is not specified, it defaults to 1. (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8)), v["arg2_value"] == -1), v["arg2_value"] == 1, False)) if n else
          If(And((Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8)), v["arg2_value"] == -1), v["arg2_value"] == 1, False))
)

def rule_31_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))

        # Constraints for rule 31
        rule_31(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

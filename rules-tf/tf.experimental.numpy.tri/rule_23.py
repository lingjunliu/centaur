import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if dtype is not specified or is invalid, N and M are small enough to be represented by default type (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 0, v["arg3_value"] == 1), v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5), v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 11), v["arg3_value"] == 12), True, And(v["arg1_value"] < 2147483647, v["arg2_value"] < 2147483647))) if n else
          If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 0, v["arg3_value"] == 1), v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5), v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 11), v["arg3_value"] == 12), True, And(v["arg1_value"] < 2147483647, v["arg2_value"] < 2147483647)))
)

def rule_23_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 23
        rule_23(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If num_rows and num_cols are both equal to a prime number, batch_shape must be 1 (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(Or(Or(v["arg2_value"] == 2, v["arg2_value"] == 3), v["arg2_value"] == 5), v["arg2_value"] == 7), v["arg2_value"] == 11)), (Or(Or(Or(Or(v["arg3_value"] == 2, v["arg3_value"] == 3), v["arg3_value"] == 5), v["arg3_value"] == 7), v["arg3_value"] == 11))), (v["arg1_length"] < 2), True)) if n else
          If(And((Or(Or(Or(Or(v["arg2_value"] == 2, v["arg2_value"] == 3), v["arg2_value"] == 5), v["arg2_value"] == 7), v["arg2_value"] == 11)), (Or(Or(Or(Or(v["arg3_value"] == 2, v["arg3_value"] == 3), v["arg3_value"] == 5), v["arg3_value"] == 7), v["arg3_value"] == 11))), (v["arg1_length"] < 2), True))
)

def rule_54_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 54
        rule_54(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

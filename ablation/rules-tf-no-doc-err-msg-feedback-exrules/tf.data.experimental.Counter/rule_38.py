import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the dtype is int8, start should be between -128 and 127, and if dtype is uint8, the start should be between 0 and 255 (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 1, And(v["arg1_value"] >= -128, v["arg1_value"] <= 127), If(v["arg2_value"] == 5, And(v["arg1_value"] >= 0, v["arg1_value"] <= 255), True))) if n else
          If(v["arg2_value"] == 1, And(v["arg1_value"] >= -128, v["arg1_value"] <= 127), If(v["arg2_value"] == 5, And(v["arg1_value"] >= 0, v["arg1_value"] <= 255), True)))
)

def rule_38_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

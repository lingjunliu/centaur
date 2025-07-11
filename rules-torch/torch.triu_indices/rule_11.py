import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If offset is negative, then absolute value of offset is less than or equal to min of row and col.  Use 0-v_3 for absolute value (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] < 0, And(0 - v["arg3_value"] <= v["arg1_value"], 0 - v["arg3_value"] <= v["arg2_value"]), False)) if n else
          If(v["arg3_value"] < 0, And(0 - v["arg3_value"] <= v["arg1_value"], 0 - v["arg3_value"] <= v["arg2_value"]), False))
)

def rule_11_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 11
        rule_11(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If deterministic is true, seed and seed2 must be specified (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == True, And(v["arg2_value"] != 0, v["arg3_value"] != 0), True)) if n else
          If(v["arg3_value"] == True, And(v["arg2_value"] != 0, v["arg3_value"] != 0), True))
)

def rule_24_func(arg3, arg2, solver=None, neg=False):
    arg3 = next(iter(arg3.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg3_value = Int('arg3_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg3_value == int(arg3))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 24
        rule_24(solver, {'arg3_value': arg3_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg3_value': arg3['value'], 'arg2_value': arg2['value']}, neg)

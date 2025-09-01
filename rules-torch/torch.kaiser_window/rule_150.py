import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If beta is a NaN value, then error, and the length of window should be less than max (Rule 150)

rule_150 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] == v["arg1_value"], v["arg2_value"] < 10000)) if n else
          And(v["arg1_value"] == v["arg1_value"], v["arg2_value"] < 10000))
)

def rule_150_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 150
        rule_150(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_150(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

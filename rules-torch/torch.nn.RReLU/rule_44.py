import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Prevent the difference between lower and upper bounds from being too large while also ensuring they relate to each other (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_value"] > v["arg1_value"] - 1e3, v["arg1_value"] < v["arg2_value"] + 1e3)) if n else
          And(v["arg2_value"] > v["arg1_value"] - 1e3, v["arg1_value"] < v["arg2_value"] + 1e3))
)

def rule_44_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)

        # Constraints for rule 44
        rule_44(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

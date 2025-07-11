import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The sum of the squares of the parameters should be bounded. (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] * v["arg1_value"] + v["arg2_value"] * v["arg2_value"] + v["arg3_value"] * v["arg3_value"] < 1e12) if n else
          v["arg1_value"] * v["arg1_value"] + v["arg2_value"] * v["arg2_value"] + v["arg3_value"] * v["arg3_value"] < 1e12)
)

def rule_46_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 46
        rule_46(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

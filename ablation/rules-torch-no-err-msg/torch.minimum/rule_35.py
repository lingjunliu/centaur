import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If both inputs are scalar numbers, they should have compatible types. (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or((And(v["arg1_value"] == True, v["arg2_value"] == 1.0)), (And(v["arg1_value"] == 1.0, v["arg2_value"] == True))), (And(v["arg1_value"] == True, v["arg2_value"] == 1))), (And(v["arg1_value"] == 1, v["arg2_value"] == True)))) if n else
          Or(Or(Or((And(v["arg1_value"] == True, v["arg2_value"] == 1.0)), (And(v["arg1_value"] == 1.0, v["arg2_value"] == True))), (And(v["arg1_value"] == True, v["arg2_value"] == 1))), (And(v["arg1_value"] == 1, v["arg2_value"] == True))))
)

def rule_35_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not (isinstance(arg2, (float, np.floating)) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 35
        rule_35(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

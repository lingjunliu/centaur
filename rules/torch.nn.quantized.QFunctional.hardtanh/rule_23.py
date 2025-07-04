import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The parameters min_val and max_val should be of type String or float (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_value"] == 6, (v["arg1_value"] + 0) == (v["arg1_value"] + 0))), (Or(v["arg2_value"] == 6, (v["arg2_value"] + 0) == (v["arg2_value"] + 0))))) if n else
          And((Or(v["arg1_value"] == 6, (v["arg1_value"] + 0) == (v["arg1_value"] + 0))), (Or(v["arg2_value"] == 6, (v["arg2_value"] + 0) == (v["arg2_value"] + 0)))))
)

def rule_23_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str) or isinstance(arg1, (float, np.floating))):
            return False
        if not (isinstance(arg2, str) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 23
        rule_23(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

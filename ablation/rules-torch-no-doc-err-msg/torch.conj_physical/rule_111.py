import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the value is less than 10, then it should be a positive, otherwise can be negative (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 10, v["arg1_value"] > 0, True)) if n else
          If(v["arg1_value"] < 10, v["arg1_value"] > 0, True))
)

def rule_111_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 111
        rule_111(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_value': arg1['value']}, neg)

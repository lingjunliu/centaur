import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# momentum must be in [0, 1 (Rule 7)

rule_7 = lambda s, v, n=False: (
    s.add(Not(And(0 <= v["arg1_value"], v["arg1_value"] < 1)) if n else
          And(0 <= v["arg1_value"], v["arg1_value"] < 1))
)

def rule_7_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 7
        rule_7(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_7(solver, {'arg1_value': arg1['value']}, neg)

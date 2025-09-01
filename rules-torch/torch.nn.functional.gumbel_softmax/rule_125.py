import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Hard, v_3 could be boolean, int or float only for edge cases like fp16 (Rule 125)

rule_125 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] == 0.0), v["arg1_value"] == 1.0)) if n else
          Or(Or(Or(v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] == 0.0), v["arg1_value"] == 1.0))
)

def rule_125_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 125
        rule_125(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_125(solver, {'arg1_value': arg1['value']}, neg)

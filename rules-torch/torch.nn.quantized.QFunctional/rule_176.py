import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Bias should be valid and between 0 and 5000 (Rule 176)

rule_176 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] < 5000.0, v["arg1_value"] >= 0)) if n else
          And(v["arg1_value"] < 5000.0, v["arg1_value"] >= 0))
)

def rule_176_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 176
        rule_176(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_176(solver, {'arg1_value': arg1['value']}, neg)

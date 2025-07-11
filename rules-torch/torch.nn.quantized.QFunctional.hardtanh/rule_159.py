import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If one bound is infinity, then the other must be STRING (representing none (Rule 159)

rule_159 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == NUMBER, v["arg2_value"] == STRING, If(v["arg2_value"] == NUMBER, v["arg1_value"] == STRING, False))) if n else
          If(v["arg1_value"] == NUMBER, v["arg2_value"] == STRING, If(v["arg2_value"] == NUMBER, v["arg1_value"] == STRING, False)))
)

def rule_159_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, (float, np.floating)) or isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 159
        rule_159(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_159(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if dimension is not none, it should be a non-negative integer (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != 6, v["arg1_value"] >= 0, False)) if n else
          If(v["arg1_value"] != 6, v["arg1_value"] >= 0, False))
)

def rule_12_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 12
        rule_12(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_value': arg1['value']}, neg)

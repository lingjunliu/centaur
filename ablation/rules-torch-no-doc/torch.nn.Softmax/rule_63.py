import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check if the dimension parameter is within the valid range of dimension (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg2_value"] == 6), If(v["arg2_value"] >= 0, v["arg2_value"] < v["arg1_ndim"], (0 - v["arg2_value"]) <= v["arg1_ndim"]))) if n else
          Or((v["arg2_value"] == 6), If(v["arg2_value"] >= 0, v["arg2_value"] < v["arg1_ndim"], (0 - v["arg2_value"]) <= v["arg1_ndim"])))
)

def rule_63_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 63
        rule_63(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If a dimension is provided ensure it is a valid dimension, otherwise allow the default behavior (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg2_value"] == 6), If(v["arg2_value"] >= 0, (v["arg2_value"] < v["arg1_ndim"]), ((0 - v["arg2_value"]) <= v["arg1_ndim"])))) if n else
          Or((v["arg2_value"] == 6), If(v["arg2_value"] >= 0, (v["arg2_value"] < v["arg1_ndim"]), ((0 - v["arg2_value"]) <= v["arg1_ndim"]))))
)

def rule_85_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 85
        rule_85(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

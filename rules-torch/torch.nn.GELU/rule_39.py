import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Approximate is either tanh OR input ndim > 0 (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg1_value"] == 12), (v["arg2_ndim"] > 0))) if n else
          Or((v["arg1_value"] == 12), (v["arg2_ndim"] > 0)))
)

def rule_39_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 39
        rule_39(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

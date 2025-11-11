import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# numeric scalar with tensor is allowed (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(v["arg1_value"] == True, v["arg1_value"] == False), (And(-1000000 <= v["arg1_value"], v["arg1_value"] <= 1000000)))), v["arg2_ndim"] >= 0)) if n else
          And((Or(Or(v["arg1_value"] == True, v["arg1_value"] == False), (And(-1000000 <= v["arg1_value"], v["arg1_value"] <= 1000000)))), v["arg2_ndim"] >= 0))
)

def rule_10_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating)) or isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 10
        rule_10(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

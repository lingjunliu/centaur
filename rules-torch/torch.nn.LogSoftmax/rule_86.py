import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# dim should be within the valid range using arithmetics and bound values (Rule 86)

rule_86 = lambda s, v, n=False: (
    s.add(Not(And(And(And((v["arg2_value"] + v["arg1_ndim"]) >= 0, (v["arg2_value"] - (v["arg1_ndim"] - 1)) <= 0), v["arg2_value"] > -2147483647), v["arg2_value"] < 2147483647)) if n else
          And(And(And((v["arg2_value"] + v["arg1_ndim"]) >= 0, (v["arg2_value"] - (v["arg1_ndim"] - 1)) <= 0), v["arg2_value"] > -2147483647), v["arg2_value"] < 2147483647))
)

def rule_86_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 86
        rule_86(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If one tensor has dimension N x M, and the other has dimension N, p must be an integer or float with no fractional part (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(If(Or((And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 1)), (And(v["arg2_ndim"] == 2, v["arg1_ndim"] == 1))), v["arg3_value"] == (v["arg3_value"] * 1.0), False)) if n else
          If(Or((And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 1)), (And(v["arg2_ndim"] == 2, v["arg1_ndim"] == 1))), v["arg3_value"] == (v["arg3_value"] * 1.0), False))
)

def rule_11_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == arg3)

        # Constraints for rule 11
        rule_11(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)

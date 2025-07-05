import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If p is not equal to 2, enforce constraint that it's lower than 100 or if it is 2 then tensor must have at least one dimension (Rule 81)

rule_81 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != 2, v["arg1_value"] < 100, v["arg2_ndim"] > 0)) if n else
          If(v["arg1_value"] != 2, v["arg1_value"] < 100, v["arg2_ndim"] > 0))
)

def rule_81_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 81
        rule_81(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_81(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

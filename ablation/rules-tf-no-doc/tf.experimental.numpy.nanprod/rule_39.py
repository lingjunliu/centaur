import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# initial value must be a scalar if provided with axis and initial (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == 6, v["arg3_value"] == 6), True, v["arg1_ndim"] == 0)) if n else
          If(Or(v["arg2_value"] == 6, v["arg3_value"] == 6), True, v["arg1_ndim"] == 0))
)

def rule_39_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, str)):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating)) or isinstance(arg3, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 39
        rule_39(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

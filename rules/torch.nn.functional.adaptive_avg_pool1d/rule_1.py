import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# input tensor should have 2 or 3 dimensions, and output size should be non-negative. (Rule 1)

rule_1 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_ndim"] == 2, v["arg1_ndim"] == 3)), v["arg2_value"] >= 0)) if n else
          And((Or(v["arg1_ndim"] == 2, v["arg1_ndim"] == 3)), v["arg2_value"] >= 0))
)

def rule_1_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 1
        rule_1(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Tensor ndim must be less than 5 or tensor dtype is string or tuple length is 3 or dtype must be floating  (Rule 120)

rule_120 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(v["arg1_ndim"] < 5, v["arg1_dtype"] == 11), v["arg2_length"] == 3), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)) if n else
          Or(Or(Or(Or(v["arg1_ndim"] < 5, v["arg1_dtype"] == 11), v["arg2_length"] == 3), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8))
)

def rule_120_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 120
        rule_120(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_120(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_length': arg2['length']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# output_size has length of 1 or 3 only if input is 3D, otherwise only can be 0, 2, or 4 (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 3, Or(v["arg2_length"] == 1, v["arg2_length"] == 3), Or(Or(v["arg2_length"] == 0, v["arg2_length"] == 2), v["arg2_length"] == 4))) if n else
          If(v["arg1_ndim"] == 3, Or(v["arg2_length"] == 1, v["arg2_length"] == 3), Or(Or(v["arg2_length"] == 0, v["arg2_length"] == 2), v["arg2_length"] == 4)))
)

def rule_39_func(arg1, arg2, solver=None, neg=False):
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
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 39
        rule_39(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)

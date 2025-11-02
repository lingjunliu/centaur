import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Tuple length is valid (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And((v["arg2_ndim"] == 3), (Or((v["arg1_length"] == 2), (v["arg1_length"] == 6))))), (And((v["arg2_ndim"] == 4), (Or(Or((v["arg1_length"] == 2), (v["arg1_length"] == 4)), (v["arg1_length"] == 8)))))), (v["arg2_ndim"] < 3))) if n else
          Or(Or((And((v["arg2_ndim"] == 3), (Or((v["arg1_length"] == 2), (v["arg1_length"] == 6))))), (And((v["arg2_ndim"] == 4), (Or(Or((v["arg1_length"] == 2), (v["arg1_length"] == 4)), (v["arg1_length"] == 8)))))), (v["arg2_ndim"] < 3)))
)

def rule_73_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 73
        rule_73(solver, {'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim']}, neg)

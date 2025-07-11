import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Suppress "IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1 (Rule 92)

rule_92 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, True, False)) if n else
          If(v["arg1_ndim"] > 0, True, False))
)

def rule_92_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 92
        rule_92(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_92(solver, {'arg1_ndim': arg1['ndim']}, neg)

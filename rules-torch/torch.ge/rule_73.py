import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check that if `other` is a tensor, the shapes are broadcastable. Simplified (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_ndim"] == v["arg2_ndim"], (And(v["arg1_ndim"] > 0, v["arg2_ndim"] == 0))), (And(v["arg1_ndim"] == 0, v["arg2_ndim"] > 0)))) if n else
          Or(Or(v["arg1_ndim"] == v["arg2_ndim"], (And(v["arg1_ndim"] > 0, v["arg2_ndim"] == 0))), (And(v["arg1_ndim"] == 0, v["arg2_ndim"] > 0))))
)

def rule_73_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 73
        rule_73(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']}, neg)

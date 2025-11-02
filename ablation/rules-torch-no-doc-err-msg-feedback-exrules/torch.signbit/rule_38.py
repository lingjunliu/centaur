import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input tensor must have a valid number of dimensions, between 0 and 8 inclusive (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] >= 0, v["arg1_ndim"] <= 8)) if n else
          And(v["arg1_ndim"] >= 0, v["arg1_ndim"] <= 8))
)

def rule_38_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 38
        rule_38(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_ndim': arg1['ndim']}, neg)

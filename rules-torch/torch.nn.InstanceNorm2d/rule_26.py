import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input tensor must be a 3D or 4D tensor to be compatible (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)) if n else
          Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4))
)

def rule_26_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 26
        rule_26(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_ndim': arg1['ndim']}, neg)

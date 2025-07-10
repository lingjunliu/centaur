import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# asset_path tensor can only be 0-D or 1-D if it is a tensor (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_ndim"] == 0, v["arg1_ndim"] == 1)) if n else
          Or(v["arg1_ndim"] == 0, v["arg1_ndim"] == 1))
)

def rule_39_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 39
        rule_39(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_ndim': arg1['ndim']}, neg)

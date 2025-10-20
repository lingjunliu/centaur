import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The valid dimension need to be inside an acceptable value, not too large (Rule 134)

rule_134 = lambda s, v, n=False: (
    s.add(Not(v["arg1_ndim"] < 12) if n else
          v["arg1_ndim"] < 12)
)

def rule_134_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 134
        rule_134(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_134(solver, {'arg1_ndim': arg1['ndim']}, neg)

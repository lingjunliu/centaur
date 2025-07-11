import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If both v1, and v2 are of type tensor, v1's dimension has to be smaller than v2's and v2's dtype has to be float32, and sum of their dimensions must be > 0 (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_ndim"] < v["arg2_ndim"], v["arg2_dtype"] == 7), v["arg1_ndim"] + v["arg2_ndim"] > 0)) if n else
          And(And(v["arg1_ndim"] < v["arg2_ndim"], v["arg2_dtype"] == 7), v["arg1_ndim"] + v["arg2_ndim"] > 0))
)

def rule_64_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 64
        rule_64(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)

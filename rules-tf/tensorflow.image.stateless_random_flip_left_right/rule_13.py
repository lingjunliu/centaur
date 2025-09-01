import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# seed must be int32 or int64, considering the image dimension (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4)), (Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4)))) if n else
          And((Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4)), (Or(v["arg1_ndim"] == 3, v["arg1_ndim"] == 4))))
)

def rule_13_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 13
        rule_13(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype']}, neg)

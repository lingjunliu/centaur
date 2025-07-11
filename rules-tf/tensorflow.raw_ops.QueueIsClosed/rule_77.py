import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Handle should be of type string with at least 1 dimension, if it's a tensor. (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_dtype"] == 11, (If(v["arg1_ndim"] > 0, v["arg1_ndim"] >= 1, False)))) if n else
          And(v["arg1_dtype"] == 11, (If(v["arg1_ndim"] > 0, v["arg1_ndim"] >= 1, False))))
)

def rule_77_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 77
        rule_77(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']}, neg)

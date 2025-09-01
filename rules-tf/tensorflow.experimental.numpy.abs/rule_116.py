import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The maximum dimension of the tensor must be a valid integer or zero (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, True, If(v["arg1_ndim"] == 1, True, If(v["arg1_ndim"] == 2, True, If(v["arg1_ndim"] == 3, True, If(v["arg1_ndim"] == 4, True, If(v["arg1_ndim"] == 5, True, If(v["arg1_ndim"] == 6, True, If(v["arg1_ndim"] == 7, True, If(v["arg1_ndim"] == 8, True, If(v["arg1_ndim"] == 9, True, If(v["arg1_ndim"] == 10, True, False)))))))))))) if n else
          If(v["arg1_ndim"] == 0, True, If(v["arg1_ndim"] == 1, True, If(v["arg1_ndim"] == 2, True, If(v["arg1_ndim"] == 3, True, If(v["arg1_ndim"] == 4, True, If(v["arg1_ndim"] == 5, True, If(v["arg1_ndim"] == 6, True, If(v["arg1_ndim"] == 7, True, If(v["arg1_ndim"] == 8, True, If(v["arg1_ndim"] == 9, True, If(v["arg1_ndim"] == 10, True, False))))))))))))
)

def rule_116_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 116
        rule_116(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_ndim': arg1['ndim']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The length of strides must be 1, N, or N+2 where N is the spatial dimension count (Rule 187)

rule_187 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And(v["arg1_ndim"] == 3, (Or(v["arg2_length"] == 1, v["arg2_length"] == 3)))), (And(v["arg1_ndim"] == 4, (Or(v["arg2_length"] == 1, (Or(v["arg2_length"] == 2, v["arg2_length"] == 4))))))), (And(v["arg1_ndim"] == 5, (Or(v["arg2_length"] == 1, (Or(v["arg2_length"] == 3, v["arg2_length"] == 5)))))))) if n else
          Or(Or((And(v["arg1_ndim"] == 3, (Or(v["arg2_length"] == 1, v["arg2_length"] == 3)))), (And(v["arg1_ndim"] == 4, (Or(v["arg2_length"] == 1, (Or(v["arg2_length"] == 2, v["arg2_length"] == 4))))))), (And(v["arg1_ndim"] == 5, (Or(v["arg2_length"] == 1, (Or(v["arg2_length"] == 3, v["arg2_length"] == 5))))))))
)

def rule_187_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 187
        rule_187(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_187(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)

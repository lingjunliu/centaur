import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If data format starts with NC, then ksize length should be 1 or N+2, otherwise 1 or N or N+2 (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_ndim"] == 3), (Or(Or((v["arg2_length"] == 1), (v["arg2_length"] == 3)), (v["arg2_length"] == 5))), If((v["arg1_ndim"] == 4), (Or(Or((v["arg2_length"] == 1), (v["arg2_length"] == 2)), (v["arg2_length"] == 4))), (Or(Or((v["arg2_length"] == 1), (v["arg2_length"] == 1)), (v["arg2_length"] == 3)))))) if n else
          If((v["arg1_ndim"] == 3), (Or(Or((v["arg2_length"] == 1), (v["arg2_length"] == 3)), (v["arg2_length"] == 5))), If((v["arg1_ndim"] == 4), (Or(Or((v["arg2_length"] == 1), (v["arg2_length"] == 2)), (v["arg2_length"] == 4))), (Or(Or((v["arg2_length"] == 1), (v["arg2_length"] == 1)), (v["arg2_length"] == 3))))))
)

def rule_31_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 31
        rule_31(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)

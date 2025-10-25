import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If input tensor rank is 5, then ksize length should be 1, 3 or 5 (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_ndim"] == 5), (Or(Or((v["arg2_length"] == 1), (v["arg2_length"] == 3)), (v["arg2_length"] == 5))), True)) if n else
          If((v["arg1_ndim"] == 5), (Or(Or((v["arg2_length"] == 1), (v["arg2_length"] == 3)), (v["arg2_length"] == 5))), True))
)

def rule_35_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 35
        rule_35(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)

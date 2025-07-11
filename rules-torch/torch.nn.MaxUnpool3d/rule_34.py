import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# output_size length must be 3 or 5 and the input dimension must accomodate (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg1_length"] == 3, v["arg2_ndim"] == 4)), (And(v["arg1_length"] == 5, v["arg2_ndim"] == 5)))) if n else
          Or((And(v["arg1_length"] == 3, v["arg2_ndim"] == 4)), (And(v["arg1_length"] == 5, v["arg2_ndim"] == 5))))
)

def rule_34_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 34
        rule_34(solver, {'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim']}, neg)

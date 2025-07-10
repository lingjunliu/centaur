import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If strides is a list, its length must be 1, N or N+2 where N is the number of spatial dimensions (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_length"] == 1, v["arg1_length"] == v["arg2_ndim"] - 2), v["arg1_length"] == v["arg2_ndim"])) if n else
          Or(Or(v["arg1_length"] == 1, v["arg1_length"] == v["arg2_ndim"] - 2), v["arg1_length"] == v["arg2_ndim"]))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
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

        # Constraints for rule 46
        rule_46(solver, {'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim']}, neg)

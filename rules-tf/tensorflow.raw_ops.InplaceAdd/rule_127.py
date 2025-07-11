import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if dtype of i is int32, x and v's rank must be greater than 0 (Rule 127)

rule_127 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_dtype"] == 3, And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0), False)) if n else
          If(v["arg3_dtype"] == 3, And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0), False))
)

def rule_127_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 127
        rule_127(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_127(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Input tensor's dtype can be only float16, float32, float64, int32 or int64 if weight tensor is specified (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If((v["arg2_ndim"] > 0), (Or(Or(Or(Or(v["arg1_dtype"] == 5, v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4)), False)) if n else
          If((v["arg2_ndim"] > 0), (Or(Or(Or(Or(v["arg1_dtype"] == 5, v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4)), False))
)

def rule_19_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 19
        rule_19(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim']}, neg)

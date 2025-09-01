import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# x's data type must be one of the allowed types, axis must be int32 or int64, axis must be a scalar, and the rank of x must be greater than 0 (Rule 91)

rule_91 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 16), v["arg1_dtype"] == 10), v["arg1_dtype"] == 6), v["arg1_dtype"] == 17), v["arg1_dtype"] == 18)), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), v["arg2_ndim"] == 0), v["arg1_ndim"] > 0)) if n else
          And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 16), v["arg1_dtype"] == 10), v["arg1_dtype"] == 6), v["arg1_dtype"] == 17), v["arg1_dtype"] == 18)), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), v["arg2_ndim"] == 0), v["arg1_ndim"] > 0))
)

def rule_91_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 91
        rule_91(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_91(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']}, neg)

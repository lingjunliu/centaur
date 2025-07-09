import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if x is np.float16, y must be np.float16 and also out must be np.float16, ndim should be same and y should not be integer (Rule 123)

rule_123 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And(And(And(v["arg2_dtype"] == 6, v["arg3_dtype"] == 6), v["arg1_ndim"] == v["arg2_ndim"]), (And(And(And(And(v["arg2_dtype"] != 1, v["arg2_dtype"] != 2), v["arg2_dtype"] != 3), v["arg2_dtype"] != 4), v["arg2_dtype"] != 5))), False)) if n else
          If(v["arg1_dtype"] == 6, And(And(And(v["arg2_dtype"] == 6, v["arg3_dtype"] == 6), v["arg1_ndim"] == v["arg2_ndim"]), (And(And(And(And(v["arg2_dtype"] != 1, v["arg2_dtype"] != 2), v["arg2_dtype"] != 3), v["arg2_dtype"] != 4), v["arg2_dtype"] != 5))), False))
)

def rule_123_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 123
        rule_123(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_123(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype']}, neg)

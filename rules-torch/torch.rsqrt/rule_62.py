import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if input is an int, then the out tensor, if specified, should be a float. This prevents the error: result type Float can't be cast to the desired output type Short (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(If((And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), v["arg2_ndim"] > 0)), (And(7 <= v["arg2_dtype"], v["arg2_dtype"] <= 9)), False)) if n else
          If((And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), v["arg2_ndim"] > 0)), (And(7 <= v["arg2_dtype"], v["arg2_dtype"] <= 9)), False))
)

def rule_62_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 62
        rule_62(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']}, neg)

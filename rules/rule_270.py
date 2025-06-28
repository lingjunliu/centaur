import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if dtype is not string and it is 2-dimensional, then tanh shouldn't be used as the string parameter. (Rule 270)

rule_270 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] != 11, v["arg1_ndim"] == 2), v["arg2_value"] != 11, False)) if n else
          If(And(v["arg1_dtype"] != 11, v["arg1_ndim"] == 2), v["arg2_value"] != 11, False))
)

def rule_270_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 270
        rule_270(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_270(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

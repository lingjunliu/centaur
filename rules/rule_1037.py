import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the string is "i,j->ij", then the tensors must be 1D and the first must have an integer data type (Rule 1037)

rule_1037 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 2, And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1), v["arg1_dtype"] >= 1), v["arg1_dtype"] <= 5), False)) if n else
          If(v["arg3_value"] == 2, And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1), v["arg1_dtype"] >= 1), v["arg1_dtype"] <= 5), False))
)

def rule_1037_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == list_of_string_values.index(arg3))

        # Constraints for rule 1037
        rule_1037(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1037(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)

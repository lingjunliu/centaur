import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input is a 5D tensor of floating type, then # channels must not be zero, and cannot be char; otherwise ndim must be at least 4 (Rule 143)

rule_143 = lambda s, v, n=False: (
    s.add(Not(And((If(v["arg1_dtype"] == 7, Select(v["arg1_shape"], 1) != 0, False)), (If(v["arg1_dtype"] != 12, v["arg1_ndim"] >= 4, False)))) if n else
          And((If(v["arg1_dtype"] == 7, Select(v["arg1_shape"], 1) != 0, False)), (If(v["arg1_dtype"] != 12, v["arg1_ndim"] >= 4, False))))
)

def rule_143_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 143
        rule_143(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_143(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)

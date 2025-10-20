import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If features tensor is bfloat16, and has more than 2 dimension, then first two dimensions must be greater than 0. (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] == 9, v["arg1_ndim"] > 2), And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), True)) if n else
          If(And(v["arg1_dtype"] == 9, v["arg1_ndim"] > 2), And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), True))
)

def rule_59_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 59
        rule_59(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)

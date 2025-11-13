import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# filter must be a 4d tensor, filter height and width must be greater than zero and depth and channel dimension should be greater than 0 and have floating type (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), v["arg1_ndim"] == 4), Select(v["arg1_shape"], 2) > 0), Select(v["arg1_shape"], 3) > 0), (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)))) if n else
          And(And(And(And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), v["arg1_ndim"] == 4), Select(v["arg1_shape"], 2) > 0), Select(v["arg1_shape"], 3) > 0), (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8))))
)

def rule_73_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 73
        rule_73(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)

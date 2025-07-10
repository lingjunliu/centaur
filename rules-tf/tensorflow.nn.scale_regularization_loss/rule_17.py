import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# regularization_loss must be a scalar tensor or have only one element with a floating-point dtype (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg1_ndim"] == 0, (If(v["arg1_dtype"] == 6, True, If(v["arg1_dtype"] == 7, True, If(v["arg1_dtype"] == 8, True, False)))))), (And(And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 1), (If(v["arg1_dtype"] == 6, True, If(v["arg1_dtype"] == 7, True, If(v["arg1_dtype"] == 8, True, False)))))))) if n else
          Or((And(v["arg1_ndim"] == 0, (If(v["arg1_dtype"] == 6, True, If(v["arg1_dtype"] == 7, True, If(v["arg1_dtype"] == 8, True, False)))))), (And(And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 1), (If(v["arg1_dtype"] == 6, True, If(v["arg1_dtype"] == 7, True, If(v["arg1_dtype"] == 8, True, False))))))))
)

def rule_17_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 17
        rule_17(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)

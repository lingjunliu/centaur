import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# LU_pivots must be a 1D tensor of int type with non-zero elements and must match rows of LU_data. (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_ndim"] == 1, (Or(Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 3), v["arg1_dtype"] == 4))), Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))) if n else
          And(And(And(v["arg1_ndim"] == 1, (Or(Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 3), v["arg1_dtype"] == 4))), Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)))
)

def rule_51_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 51
        rule_51(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape']}, neg)

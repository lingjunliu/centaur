import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if out is provided, it should have the same shape as the broadcasted shape of input and other (Rule 4)

rule_4 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0)), (And(v["arg1_ndim"] == Select(v["arg3_shape"], 0), v["arg2_ndim"] == Select(v["arg3_shape"], 0))))) if n else
          Or((And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0)), (And(v["arg1_ndim"] == Select(v["arg3_shape"], 0), v["arg2_ndim"] == Select(v["arg3_shape"], 0)))))
)

def rule_4_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 4
        rule_4(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_4(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape']}, neg)

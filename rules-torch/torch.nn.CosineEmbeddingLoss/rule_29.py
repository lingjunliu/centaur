import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If target is 1D, and one of the inputs is 2D while the other is 1D, sizes of the first dimension on 2D tensor and the 1D tensor should match (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If((And(And(v["arg3_ndim"] == 1, (Or(v["arg1_ndim"] == 2, v["arg2_ndim"] == 2))), (Or(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1)))), (If(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], 0))), True)) if n else
          If((And(And(v["arg3_ndim"] == 1, (Or(v["arg1_ndim"] == 2, v["arg2_ndim"] == 2))), (Or(v["arg1_ndim"] == 1, v["arg2_ndim"] == 1)))), (If(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], 0))), True))
)

def rule_29_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 29
        rule_29(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_ndim': arg3['ndim']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If rank of ref is one then index tensor must have the shape [:, 1] (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 1, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 1, False)) if n else
          If(v["arg1_ndim"] == 1, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 1, False))
)

def rule_44_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 44
        rule_44(solver, {'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)

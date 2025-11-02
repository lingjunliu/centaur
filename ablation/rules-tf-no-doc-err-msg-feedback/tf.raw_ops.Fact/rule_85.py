import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# shape of the first dimension of the tensor v_1 must equal the shape of the last dimension of the tensor v_2 (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], v["arg2_ndim"] - 1)) if n else
          Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], v["arg2_ndim"] - 1))
)

def rule_85_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 85
        rule_85(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dimension of second tensor is less than 5 and shape of first tensor is less than 100 (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] < 5, Select(v["arg1_shape"], 0) < 100, True)) if n else
          If(v["arg2_ndim"] < 5, Select(v["arg1_shape"], 0) < 100, True))
)

def rule_76_func(arg1, arg2, solver=None, neg=False):
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

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 76
        rule_76(solver, {'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim']}, neg)

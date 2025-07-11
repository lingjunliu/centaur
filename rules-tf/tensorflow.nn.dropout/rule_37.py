import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# noise_shape must have the same length as x if x dimension is greater than noise_shape dimension  (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 1, If(v["arg1_ndim"] > Select(v["arg2_shape"], 0), Select(v["arg2_shape"], 0) == v["arg1_ndim"], False), False)) if n else
          If(v["arg2_ndim"] == 1, If(v["arg1_ndim"] > Select(v["arg2_shape"], 0), Select(v["arg2_shape"], 0) == v["arg1_ndim"], False), False))
)

def rule_37_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 37
        rule_37(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

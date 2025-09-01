import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# delta must be in the interval [-1, 1] and Image has 3 channels in the last dimension, tensor rank >= 3 (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] >= -1.0, v["arg1_value"] <= 1.0), Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 3), v["arg2_ndim"] >= 3)) if n else
          And(And(And(v["arg1_value"] >= -1.0, v["arg1_value"] <= 1.0), Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 3), v["arg2_ndim"] >= 3))
)

def rule_30_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 30
        rule_30(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The channel dimension of the input tensor must have 3 channels for RGB images under both formats (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_ndim"] >= 3, (Or((And(v["arg1_value"] == 24, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 3)), (And(v["arg1_value"] == 25, Select(v["arg2_shape"], v["arg2_ndim"] - 3) == 3)))))) if n else
          And(v["arg2_ndim"] >= 3, (Or((And(v["arg1_value"] == 24, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 3)), (And(v["arg1_value"] == 25, Select(v["arg2_shape"], v["arg2_ndim"] - 3) == 3))))))
)

def rule_43_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 43
        rule_43(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

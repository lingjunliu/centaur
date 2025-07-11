import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# index_type should be compatible with the tensor size (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 2, Select(v["arg2_shape"], v["arg2_ndim"] - 1) <= 32767, If(v["arg1_value"] == 3, Select(v["arg2_shape"], v["arg2_ndim"] - 1) <= 2147483647, False))) if n else
          If(v["arg1_value"] == 2, Select(v["arg2_shape"], v["arg2_ndim"] - 1) <= 32767, If(v["arg1_value"] == 3, Select(v["arg2_shape"], v["arg2_ndim"] - 1) <= 2147483647, False)))
)

def rule_60_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 60
        rule_60(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

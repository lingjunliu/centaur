import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If output dtype is int32 then num_samples * batch_size must be less than max int, otherwise if output dtype is int64 then num_samples * batch_size must be less than max long, otherwise false (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 3, v["arg1_value"] * Select(v["arg2_shape"], 0) <= 2147483647, If(v["arg3_value"] == 4, v["arg1_value"] * Select(v["arg2_shape"], 0) <= 9223372036854775807, False))) if n else
          If(v["arg3_value"] == 3, v["arg1_value"] * Select(v["arg2_shape"], 0) <= 2147483647, If(v["arg3_value"] == 4, v["arg1_value"] * Select(v["arg2_shape"], 0) <= 9223372036854775807, False)))
)

def rule_116_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 116
        rule_116(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

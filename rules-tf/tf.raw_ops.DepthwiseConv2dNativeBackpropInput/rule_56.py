import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# filter and out_backprop tensors' channels/depth dimensions should be compatible with data_format and each other, and their data types should be the same (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(And((If(v["arg2_value"] == 33, Select(v["arg1_shape"], 2) == Select(v["arg3_shape"], 3), If(v["arg2_value"] == 34, Select(v["arg1_shape"], 2) == Select(v["arg3_shape"], 1), True))), v["arg1_dtype"] == v["arg3_dtype"])) if n else
          And((If(v["arg2_value"] == 33, Select(v["arg1_shape"], 2) == Select(v["arg3_shape"], 3), If(v["arg2_value"] == 34, Select(v["arg1_shape"], 2) == Select(v["arg3_shape"], 1), True))), v["arg1_dtype"] == v["arg3_dtype"]))
)

def rule_56_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 56
        rule_56(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# filter and out_backprop tensors' channels/depth dimensions should be compatible with data_format and each other, and their data types should be the same, and filter must have a float type and out_backprop must be a 4d tensor and its shape must be positive. All tensors must be 4d and input size must be a tensor of length 4 with int32 dtype. (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And((If(v["arg2_value"] == 33, Select(v["arg1_shape"], 2) == Select(v["arg3_shape"], 3), If(v["arg2_value"] == 34, Select(v["arg1_shape"], 2) == Select(v["arg3_shape"], 1), True))), v["arg1_dtype"] == v["arg3_dtype"]), (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8))), v["arg3_ndim"] == 4), And([Implies(i < (3 + 1), And(And(And(And(Select(v["arg3_shape"], i) > 0, v["arg1_ndim"] == 4), v["arg4_ndim"] == 1), Select(v["arg4_shape"], 0) == 4), v["arg4_dtype"] == 3)) for i in range(6)]))) if n else
          And(And(And(And((If(v["arg2_value"] == 33, Select(v["arg1_shape"], 2) == Select(v["arg3_shape"], 3), If(v["arg2_value"] == 34, Select(v["arg1_shape"], 2) == Select(v["arg3_shape"], 1), True))), v["arg1_dtype"] == v["arg3_dtype"]), (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8))), v["arg3_ndim"] == 4), And([Implies(i < (3 + 1), And(And(And(And(Select(v["arg3_shape"], i) > 0, v["arg1_ndim"] == 4), v["arg4_ndim"] == 1), Select(v["arg4_shape"], 0) == 4), v["arg4_dtype"] == 3)) for i in range(6)])))
)

def rule_85_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 85
        rule_85(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_dtype': arg4_dtype, 'arg4_shape': arg4_shape, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_dtype': arg4['dtype'], 'arg4_shape': arg4['shape'], 'arg4_ndim': arg4['ndim']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If seq_lengths are given, its length must match the batch_axis dimension of input and they must be int32 or int64 tensors and seq, batch axis are valid and distinct and input must have rank > 0  (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 0, And(And(And(And(And(And(And(And(And(v["arg1_ndim"] > 0, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg4_value"])), (Or((v["arg2_dtype"] == 3), (v["arg2_dtype"] == 4)))), (v["arg3_value"] >= -2147483648)), (v["arg3_value"] <= 2147483647)), (v["arg4_value"] >= -2147483648)), (v["arg4_value"] <= 2147483647)), (And(v["arg3_value"] >= (-1 * v["arg1_ndim"]), v["arg3_value"] < v["arg1_ndim"]))), (And(v["arg4_value"] >= (-1 * v["arg1_ndim"]), v["arg4_value"] < v["arg1_ndim"]))), (v["arg3_value"] != v["arg4_value"])), True)) if n else
          If(v["arg2_ndim"] > 0, And(And(And(And(And(And(And(And(And(v["arg1_ndim"] > 0, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg4_value"])), (Or((v["arg2_dtype"] == 3), (v["arg2_dtype"] == 4)))), (v["arg3_value"] >= -2147483648)), (v["arg3_value"] <= 2147483647)), (v["arg4_value"] >= -2147483648)), (v["arg4_value"] <= 2147483647)), (And(v["arg3_value"] >= (-1 * v["arg1_ndim"]), v["arg3_value"] < v["arg1_ndim"]))), (And(v["arg4_value"] >= (-1 * v["arg1_ndim"]), v["arg4_value"] < v["arg1_ndim"]))), (v["arg3_value"] != v["arg4_value"])), True))
)

def rule_46_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 46
        rule_46(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

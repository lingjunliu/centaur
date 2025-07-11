import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the specified dtype is int32 or int64, the input dtype must be integer, floating point or complex, the input tensor must have values inside a valid range and the specified dtype is not boolean or string, and the dim must be within range and the output should have the same shape as the input if no dtype is specified and if out tensor is given the dtype of the out tensor must be non boolean. (Rule 86)

rule_86 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == 3, v["arg2_value"] == 4), And(And(And(And(And(And(And(And(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), Select(v["arg1_range"], 0) > -10000), Select(v["arg1_range"], 1) < 10000), v["arg2_value"] != 0), v["arg2_value"] != 11), v["arg3_value"] >= (0 - v["arg1_ndim"])), v["arg3_value"] < v["arg1_ndim"]), (If(v["arg2_value"] == 12, And(v["arg1_ndim"] == v["arg4_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i)) for i in range(6)])), False))), v["arg4_dtype"] != 0), False)) if n else
          If(Or(v["arg2_value"] == 3, v["arg2_value"] == 4), And(And(And(And(And(And(And(And(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), v["arg1_dtype"] == 6), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), Select(v["arg1_range"], 0) > -10000), Select(v["arg1_range"], 1) < 10000), v["arg2_value"] != 0), v["arg2_value"] != 11), v["arg3_value"] >= (0 - v["arg1_ndim"])), v["arg3_value"] < v["arg1_ndim"]), (If(v["arg2_value"] == 12, And(v["arg1_ndim"] == v["arg4_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg4_shape"], i)) for i in range(6)])), False))), v["arg4_dtype"] != 0), False))
)

def rule_86_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 86
        rule_86(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim, 'arg4_dtype': arg4_dtype, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim'], 'arg4_dtype': arg4['dtype'], 'arg4_shape': arg4['shape']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# biases must have a shape compatible with the number of classes and biases must be a tensor with dimension of 1 and have the same dtype as weights. Also, the weights should have at least 2 dimensions. The shapes of weights and inputs should be compatible and greater than zero. Num Classes should be greater than zero. Dtype of biases must be float type. (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) == v["arg2_value"], v["arg1_ndim"] == 1), v["arg1_dtype"] == v["arg3_dtype"]), v["arg3_ndim"] >= 2), Select(v["arg3_shape"], 1) == Select(v["arg4_shape"], 1)), Select(v["arg3_shape"], 0) > 0), Select(v["arg4_shape"], 0) > 0), v["arg2_value"] > 0), (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8)))) if n else
          And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) == v["arg2_value"], v["arg1_ndim"] == 1), v["arg1_dtype"] == v["arg3_dtype"]), v["arg3_ndim"] >= 2), Select(v["arg3_shape"], 1) == Select(v["arg4_shape"], 1)), Select(v["arg3_shape"], 0) > 0), Select(v["arg4_shape"], 0) > 0), v["arg2_value"] > 0), (Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8))))
)

def rule_57_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
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
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 57
        rule_57(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_shape': arg4['shape']}, neg)

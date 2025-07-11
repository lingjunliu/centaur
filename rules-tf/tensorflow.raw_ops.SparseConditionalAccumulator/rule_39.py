import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The shape of tensor must match specified shape if dtype is float (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), Select(v["arg1_shape"], 0) == Select(v["arg3_shape"], 0), False)) if n else
          If(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), Select(v["arg1_shape"], 0) == Select(v["arg3_shape"], 0), False))
)

def rule_39_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 39
        rule_39(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)

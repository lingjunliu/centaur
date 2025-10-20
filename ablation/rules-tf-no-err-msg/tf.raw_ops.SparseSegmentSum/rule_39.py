import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If indices are int32, the minimum value should not be less than 0 and the maximum value should be less than shape of data at axis 0 (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 3, And(Select(v["arg3_range"], 0) >= 0, Select(v["arg3_range"], 1) < Select(v["arg2_shape"], 0)), True)) if n else
          If(v["arg1_value"] == 3, And(Select(v["arg3_range"], 0) >= 0, Select(v["arg3_range"], 1) < Select(v["arg2_shape"], 0)), True))
)

def rule_39_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 39
        rule_39(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_range': arg3['range']}, neg)

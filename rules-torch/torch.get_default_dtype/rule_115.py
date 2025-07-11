import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the default type is float32, the shape must be greater or equal than one and less than 10 (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 7, And(Select(v["arg2_shape"], 0) >= 1, Select(v["arg2_shape"], 0) <= 10), False)) if n else
          If(v["arg1_value"] == 7, And(Select(v["arg2_shape"], 0) >= 1, Select(v["arg2_shape"], 0) <= 10), False))
)

def rule_115_func(arg1, arg2, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 115
        rule_115(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)

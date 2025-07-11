import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if src_format is equal to dst_format, then the input tensor will be returned without any changes. (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == v["arg3_value"], Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], 0), False)) if n else
          If(v["arg2_value"] == v["arg3_value"], Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], 0), False))
)

def rule_87_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 87
        rule_87(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

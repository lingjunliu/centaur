import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When data format is NCHW or NHWC, dimensions 0 and 3 or 1 respectively need to be one (Rule 123)

rule_123 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 34, Select(v["arg1_shape"], 0) == 1, If(v["arg2_value"] == 33, Select(v["arg1_shape"], 3) == 1, True))) if n else
          If(v["arg2_value"] == 34, Select(v["arg1_shape"], 0) == 1, If(v["arg2_value"] == 33, Select(v["arg1_shape"], 3) == 1, True)))
)

def rule_123_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 123
        rule_123(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_123(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)

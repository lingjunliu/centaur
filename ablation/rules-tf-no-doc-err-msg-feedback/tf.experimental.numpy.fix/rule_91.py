import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# v_1 is a string and v_2 is a tensor, if v_1 represents the type of padding and it is "valid", shape(v_2,1 (Rule 91)

rule_91 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And(v["arg1_value"] == 21, Select(v["arg2_shape"], 1) > 0)), (And(v["arg1_value"] == 22, Select(v["arg2_shape"], 1) > 0))), (And(v["arg1_value"] == 23, Select(v["arg2_shape"], 1) > 0)))) if n else
          Or(Or((And(v["arg1_value"] == 21, Select(v["arg2_shape"], 1) > 0)), (And(v["arg1_value"] == 22, Select(v["arg2_shape"], 1) > 0))), (And(v["arg1_value"] == 23, Select(v["arg2_shape"], 1) > 0))))
)

def rule_91_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 91
        rule_91(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_91(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)

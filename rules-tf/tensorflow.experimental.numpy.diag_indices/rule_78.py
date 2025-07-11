import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# String equal to something and another shape needs to satisfy some condition (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] == 8, Select(v["arg2_shape"], 0) > 10)) if n else
          And(v["arg1_value"] == 8, Select(v["arg2_shape"], 0) > 10))
)

def rule_78_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 78
        rule_78(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)

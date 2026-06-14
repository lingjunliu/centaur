import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Number of channels in the filter should equal the number of channels in the input depending on the format and filter's channel_multiplier > 0 (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(And((If(v["arg3_value"] == 31, Select(v["arg1_shape"], 3) == Select(v["arg2_shape"], 2), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 2))), Select(v["arg2_shape"], 3) > 0)) if n else
          And((If(v["arg3_value"] == 31, Select(v["arg1_shape"], 3) == Select(v["arg2_shape"], 2), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 2))), Select(v["arg2_shape"], 3) > 0))
)

def rule_102_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 102
        rule_102(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

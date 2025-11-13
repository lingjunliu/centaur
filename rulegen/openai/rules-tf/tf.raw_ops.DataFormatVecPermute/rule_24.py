import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# 1D input size for channel aliases is 4 or 2 (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If((And((Or(v["arg1_value"] == 24, v["arg1_value"] == 25)), (Or(v["arg2_value"] == 24, v["arg2_value"] == 25)))), If(v["arg3_ndim"] == 1, (Or(Select(v["arg3_shape"], 0) == 4, Select(v["arg3_shape"], 0) == 2)), True), True)) if n else
          If((And((Or(v["arg1_value"] == 24, v["arg1_value"] == 25)), (Or(v["arg2_value"] == 24, v["arg2_value"] == 25)))), If(v["arg3_ndim"] == 1, (Or(Select(v["arg3_shape"], 0) == 4, Select(v["arg3_shape"], 0) == 2)), True), True))
)

def rule_24_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = String('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 24
        rule_24(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The channel value can not exceed 1000. Otherwise, alpha and beta must have small values and depth_radius needs to be less than 5. Bias should be reasonable, and the tensor needs at least 10 channels. All dimension except channels, must have a value exceeding one (Rule 142)

rule_142 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 3) > 1000, And(And(And(And(And(v["arg2_value"] < 3, v["arg3_value"] < 0.75), v["arg4_value"] < 5), v["arg5_value"] < 1000), Select(v["arg1_shape"], 3) > 10), (Or(Or(Select(v["arg1_shape"], 0) > 1, Select(v["arg1_shape"], 1) > 1), Select(v["arg1_shape"], 2) > 1))), True)) if n else
          If(Select(v["arg1_shape"], 3) > 1000, And(And(And(And(And(v["arg2_value"] < 3, v["arg3_value"] < 0.75), v["arg4_value"] < 5), v["arg5_value"] < 1000), Select(v["arg1_shape"], 3) > 10), (Or(Or(Select(v["arg1_shape"], 0) > 1, Select(v["arg1_shape"], 1) > 1), Select(v["arg1_shape"], 2) > 1))), True))
)

def rule_142_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not isinstance(arg5, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Real('arg5_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == arg5)

        # Constraints for rule 142
        rule_142(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_142(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When the input image size is small and depth_radius is large, alpha or beta should be small enough (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_shape"], 1) < 20, v["arg2_value"] > 5), Or(v["arg3_value"] < 100, v["arg4_value"] < 5), False)) if n else
          If(And(Select(v["arg1_shape"], 1) < 20, v["arg2_value"] > 5), Or(v["arg3_value"] < 100, v["arg4_value"] < 5), False))
)

def rule_47_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_value = Real('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)

        # Constraints for rule 47
        rule_47(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If float v_1 is greater than 0 and equal or smaller to float v_2 and also dimension of v_3 tensor is equal to 2, then shape of 0 dimension of tensor v_3 must not be same with its shape in 1 dimension (Rule 531)

rule_531 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] > 0, v["arg1_value"] <= v["arg2_value"]), v["arg3_ndim"] == 2), Select(v["arg3_shape"], 0) != Select(v["arg3_shape"], 1), False)) if n else
          If(And(And(v["arg1_value"] > 0, v["arg1_value"] <= v["arg2_value"]), v["arg3_ndim"] == 2), Select(v["arg3_shape"], 0) != Select(v["arg3_shape"], 1), False))
)

def rule_531_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 531
        rule_531(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_531(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)

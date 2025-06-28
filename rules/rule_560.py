import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If string v_1 is equal to tanh and float v_2 is greater than 1, then at least one shape for tensor v_3 has to be greater than 0. (Rule 560)

rule_560 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 11, v["arg2_value"] > 1), Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]), False)) if n else
          If(And(v["arg1_value"] == 11, v["arg2_value"] > 1), Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]), False))
)

def rule_560_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 560
        rule_560(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_560(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)

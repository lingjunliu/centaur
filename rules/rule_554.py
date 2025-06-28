import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if float v_1 is between -1 and 1, and v_2 string does not equal to none, then there has to exists one dimension of tensor v_3 such that is shape is equal to int 1 (Rule 554)

rule_554 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_value"] >= -1, v["arg1_value"] <= 1), v["arg2_value"] != 6), v["arg3_ndim"] > 0), Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == 1) for i in range(6)]), False)) if n else
          If(And(And(And(v["arg1_value"] >= -1, v["arg1_value"] <= 1), v["arg2_value"] != 6), v["arg3_ndim"] > 0), Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) == 1) for i in range(6)]), False))
)

def rule_554_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = String('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_string_values.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 554
        rule_554(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_554(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)

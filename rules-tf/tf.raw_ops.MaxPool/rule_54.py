import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If padding is EXPLICIT and data format is NHWC, then the explicit padding values must be less than half the input dimensions, except for the batch and channel. (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 30, v["arg3_ndim"] == 4), And(And(And(Select(v["arg2_values"], 2) < Select(v["arg3_shape"], 1) / 2, Select(v["arg2_values"], 3) < Select(v["arg3_shape"], 1) / 2), Select(v["arg2_values"], 4) < Select(v["arg3_shape"], 2) / 2), Select(v["arg2_values"], 5) < Select(v["arg3_shape"], 2) / 2), True)) if n else
          If(And(v["arg1_value"] == 30, v["arg3_ndim"] == 4), And(And(And(Select(v["arg2_values"], 2) < Select(v["arg3_shape"], 1) / 2, Select(v["arg2_values"], 3) < Select(v["arg3_shape"], 1) / 2), Select(v["arg2_values"], 4) < Select(v["arg3_shape"], 2) / 2), Select(v["arg2_values"], 5) < Select(v["arg3_shape"], 2) / 2), True))
)

def rule_54_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 54
        rule_54(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)

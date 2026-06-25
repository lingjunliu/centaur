import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# General parameter validation (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(v["arg1_value"] == 33, (Or(v["arg1_value"] == 28, v["arg1_value"] == 29))), v["arg2_ndim"] == 3), v["arg3_ndim"] == 4), Select(v["arg3_shape"], 0) > 0), Select(v["arg3_shape"], 1) > 0), Select(v["arg3_shape"], 2) > 0), Select(v["arg3_shape"], 3) > 0), v["arg4_length"] == 4), v["arg5_length"] == 4)) if n else
          And(And(And(And(And(And(And(And(And(v["arg1_value"] == 33, (Or(v["arg1_value"] == 28, v["arg1_value"] == 29))), v["arg2_ndim"] == 3), v["arg3_ndim"] == 4), Select(v["arg3_shape"], 0) > 0), Select(v["arg3_shape"], 1) > 0), Select(v["arg3_shape"], 2) > 0), Select(v["arg3_shape"], 3) > 0), v["arg4_length"] == 4), v["arg5_length"] == 4))
)

def rule_99_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg5_length = Int('arg5_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_length == len(arg4))
        solver.add(arg5_length == len(arg5))

        # Constraints for rule 99
        rule_99(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_length': arg4_length, 'arg5_length': arg5_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_length': arg4['length'], 'arg5_length': arg5['length']}, neg)

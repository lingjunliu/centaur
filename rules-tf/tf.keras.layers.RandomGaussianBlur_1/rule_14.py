import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# channels_first format must have positive channel dimensions for 3D and 4D inputs (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(And((If(And(v["arg1_value"] == 25, v["arg2_ndim"] == 3), Select(v["arg2_shape"], 0) > 0, True)), (If(And(v["arg1_value"] == 25, v["arg2_ndim"] == 4), Select(v["arg2_shape"], 1) > 0, True)))) if n else
          And((If(And(v["arg1_value"] == 25, v["arg2_ndim"] == 3), Select(v["arg2_shape"], 0) > 0, True)), (If(And(v["arg1_value"] == 25, v["arg2_ndim"] == 4), Select(v["arg2_shape"], 1) > 0, True))))
)

def rule_14_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 14
        rule_14(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If data_format is channels_first and input tensor is rank 3, output shape is [batch, 2, height, width] (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == 25, v["arg1_ndim"] == 3), And(And(And(Select(v["arg3_shape"], 0) == Select(v["arg1_shape"], 0), Select(v["arg3_shape"], 1) == 2), Select(v["arg3_shape"], 2) == Select(v["arg1_shape"], 2)), Select(v["arg3_shape"], 3) == Select(v["arg1_shape"], 3)), True)) if n else
          If(And(v["arg2_value"] == 25, v["arg1_ndim"] == 3), And(And(And(Select(v["arg3_shape"], 0) == Select(v["arg1_shape"], 0), Select(v["arg3_shape"], 1) == 2), Select(v["arg3_shape"], 2) == Select(v["arg1_shape"], 2)), Select(v["arg3_shape"], 3) == Select(v["arg1_shape"], 3)), True))
)

def rule_21_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 21
        rule_21(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)

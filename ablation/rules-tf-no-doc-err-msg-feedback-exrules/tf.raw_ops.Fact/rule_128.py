import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the string is `softmax`, then the input tensor must have at least 2 dimension and the type code has to be 7 (float32 (Rule 128)

rule_128 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 14, And(v["arg2_ndim"] >= 2, v["arg3_value"] == 7), True)) if n else
          If(v["arg1_value"] == 14, And(v["arg2_ndim"] >= 2, v["arg3_value"] == 7), True))
)

def rule_128_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 128
        rule_128(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_128(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)

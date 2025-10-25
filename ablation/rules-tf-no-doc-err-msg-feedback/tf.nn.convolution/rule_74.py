import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The number of dimensions in strides must match the number of spatial dimensions in the input. (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 25, v["arg1_length"] == v["arg3_ndim"] - 2, If(v["arg2_value"] == 24, v["arg1_length"] == v["arg3_ndim"] - 2, True))) if n else
          If(v["arg2_value"] == 25, v["arg1_length"] == v["arg3_ndim"] - 2, If(v["arg2_value"] == 24, v["arg1_length"] == v["arg3_ndim"] - 2, True)))
)

def rule_74_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = String('arg2_value')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 74
        rule_74(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)

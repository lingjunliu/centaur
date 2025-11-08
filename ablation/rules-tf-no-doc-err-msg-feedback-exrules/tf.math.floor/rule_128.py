import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

#  if integer 1 is smaller than 10, then tensor shape should be less than tuple of integer length (Rule 128)

rule_128 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 10, v["arg2_ndim"] < v["arg3_length"], True)) if n else
          If(v["arg1_value"] < 10, v["arg2_ndim"] < v["arg3_length"], True))
)

def rule_128_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 128
        rule_128(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_128(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_length': arg3['length']}, neg)

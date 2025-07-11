import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If verify_shape is true then shape of values must be equal to the argument shape (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, v["arg2_ndim"] == v["arg3_length"], False)) if n else
          If(v["arg1_value"] == True, v["arg2_ndim"] == v["arg3_length"], False))
)

def rule_31_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 31
        rule_31(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_length': arg3['length']}, neg)

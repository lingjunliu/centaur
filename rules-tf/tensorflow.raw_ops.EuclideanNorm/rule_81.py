import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if keep_dims is false and axis is a tuple, then output ndim should be equal to the input ndim minus the number of reduction dimensions, unless it's zero. (Rule 81)

rule_81 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == False, v["arg1_ndim"] == v["arg3_ndim"] + v["arg4_length"], False)) if n else
          If(v["arg2_value"] == False, v["arg1_ndim"] == v["arg3_ndim"] + v["arg4_length"], False))
)

def rule_81_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg4_length = Int('arg4_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_length == len(arg4))

        # Constraints for rule 81
        rule_81(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_81(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg4_length': arg4['length']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the string value is 'mean', then the tensor's dimensions must be same as the list of int's length + 1  (Rule 988)

rule_988 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 7, v["arg1_ndim"] == v["arg3_length"] + 1, False)) if n else
          If(v["arg2_value"] == 7, v["arg1_ndim"] == v["arg3_length"] + 1, False))
)

def rule_988_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = String('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_string_values.index(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 988
        rule_988(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_988(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)

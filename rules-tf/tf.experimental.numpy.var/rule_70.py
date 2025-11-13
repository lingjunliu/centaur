import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the output tensor is specified then its rank must be equal to the rank of input minus the length of axis if the axis is specified as a tuple and keepdims is false (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] == False, v["arg3_ndim"] == v["arg1_ndim"] - v["arg2_length"], True)) if n else
          If(v["arg4_value"] == False, v["arg3_ndim"] == v["arg1_ndim"] - v["arg2_length"], True))
)

def rule_70_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg3_ndim = Int('arg3_ndim')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_value == arg4)

        # Constraints for rule 70
        rule_70(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)

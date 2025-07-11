import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# frame_length must be less than the signal length when pad_end is false (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"], True, v["arg1_value"] < Select(v["arg3_shape"], v["arg2_value"]))) if n else
          If(v["arg4_value"], True, v["arg1_value"] < Select(v["arg3_shape"], v["arg2_value"])))
)

def rule_21_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == arg4)

        # Constraints for rule 21
        rule_21(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)

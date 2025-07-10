import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# the shape of tensor v_1 at dimension v_2 must be equal to shape of tensor v_3 at dimension v_4 (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_shape"], v["arg2_value"]) == Select(v["arg3_shape"], v["arg4_value"])) if n else
          Select(v["arg1_shape"], v["arg2_value"]) == Select(v["arg3_shape"], v["arg4_value"]))
)

def rule_12_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Int('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 12
        rule_12(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)

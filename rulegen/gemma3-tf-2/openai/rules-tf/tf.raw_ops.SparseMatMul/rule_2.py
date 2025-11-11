import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# matmul dimension compatibility considering transposes (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"], (If(v["arg4_value"], Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), (If(v["arg4_value"], Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 0))))) if n else
          If(v["arg3_value"], (If(v["arg4_value"], Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), (If(v["arg4_value"], Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 1), Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 0)))))
)

def rule_2_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)

        # Constraints for rule 2
        rule_2(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

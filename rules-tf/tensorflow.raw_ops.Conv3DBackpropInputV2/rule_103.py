import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if shape filter and input size depth are equal, dilation must equal one (Rule 103)

rule_103 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 1), Select(v["arg3_values"], 1) == 1, False)) if n else
          If(Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 1), Select(v["arg3_values"], 1) == 1, False))
)

def rule_103_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 103
        rule_103(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values']}, neg)

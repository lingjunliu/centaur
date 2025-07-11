import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# beta must be less than threshold, taking into consideration the size of input tensor (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) > 1000, v["arg2_value"] <= v["arg3_value"], False)) if n else
          If(Select(v["arg1_shape"], 0) > 1000, v["arg2_value"] <= v["arg3_value"], False))
)

def rule_53_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 53
        rule_53(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

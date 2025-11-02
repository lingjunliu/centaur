import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If num_features is specified, it must match the size of the weight tensor's first dimension. (Rule 89)

rule_89 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != 6, Select(v["arg2_shape"], 0) == v["arg1_value"], True)) if n else
          If(v["arg1_value"] != 6, Select(v["arg2_shape"], 0) == v["arg1_value"], True))
)

def rule_89_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, str)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 89
        rule_89(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_89(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)

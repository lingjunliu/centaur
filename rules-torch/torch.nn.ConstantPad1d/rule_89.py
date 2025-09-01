import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Tuple element count is length 2, and smaller than max dimension size of the tensor itself (Rule 89)

rule_89 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_length"] == 2, Select(v["arg1_values"], 0) >= 0), Select(v["arg1_values"], 0) < Select(v["arg2_shape"], v["arg2_ndim"] - 1)), Select(v["arg1_values"], 1) >= 0), Select(v["arg1_values"], 1) < Select(v["arg2_shape"], v["arg2_ndim"] - 1))) if n else
          And(And(And(And(v["arg1_length"] == 2, Select(v["arg1_values"], 0) >= 0), Select(v["arg1_values"], 0) < Select(v["arg2_shape"], v["arg2_ndim"] - 1)), Select(v["arg1_values"], 1) >= 0), Select(v["arg1_values"], 1) < Select(v["arg2_shape"], v["arg2_ndim"] - 1)))
)

def rule_89_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 89
        rule_89(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_89(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

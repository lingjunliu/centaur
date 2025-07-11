import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if indices is provided, their values should be within the valid range (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 1, (And(Select(v["arg2_range"], 0) >= 0, Select(v["arg2_range"], 1) < Select(v["arg1_shape"], 0))), False)) if n else
          If(v["arg2_ndim"] == 1, (And(Select(v["arg2_range"], 0) >= 0, Select(v["arg2_range"], 1) < Select(v["arg1_shape"], 0))), False))
)

def rule_59_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 59
        rule_59(solver, {'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)

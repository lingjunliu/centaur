import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# For 3-dimensional tensor v_1, if sum of its first dimension's shape and second dimension's shape is greater than 100, then bool v_2 should be false (Rule 364)

rule_364 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 3, If(Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], 1) > 100, False, v["arg2_value"]), False)) if n else
          If(v["arg1_ndim"] == 3, If(Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], 1) > 100, False, v["arg2_value"]), False))
)

def rule_364_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)

        # Constraints for rule 364
        rule_364(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_364(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

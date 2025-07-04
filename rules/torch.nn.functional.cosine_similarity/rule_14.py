import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if x1 and x2 have the same number of dimensions, then at least one of the dimensions must be greater than 1, excluding the dimension along which the cosine similarity is computed. (Rule 14)

rule_14 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == v["arg2_ndim"], (Or([And(i < (v["arg1_ndim"] - 1 + 1), And(i != v["arg3_value"], Select(v["arg1_shape"], i) > 1)) for i in range(6)])))) if n else
          And(v["arg1_ndim"] == v["arg2_ndim"], (Or([And(i < (v["arg1_ndim"] - 1 + 1), And(i != v["arg3_value"], Select(v["arg1_shape"], i) > 1)) for i in range(6)]))))
)

def rule_14_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 14
        rule_14(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_14(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)

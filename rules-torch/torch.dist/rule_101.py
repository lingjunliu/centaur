import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If neither tensor is a scalar, then shapes must have at least one dimension greater than 0 and number of elements >0 (Rule 101)

rule_101 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0)), (And(And(And(And((And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0)), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), (Or([And(j < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], j) > 0) for j in range(6)]))), (Select(v["arg1_range"], 0) < Select(v["arg1_range"], 1))), (Select(v["arg2_range"], 0) < Select(v["arg2_range"], 1)))))) if n else
          Or((And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0)), (And(And(And(And((And(v["arg1_ndim"] > 0, v["arg2_ndim"] > 0)), (Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))), (Or([And(j < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], j) > 0) for j in range(6)]))), (Select(v["arg1_range"], 0) < Select(v["arg1_range"], 1))), (Select(v["arg2_range"], 0) < Select(v["arg2_range"], 1))))))
)

def rule_101_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 101
        rule_101(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_101(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

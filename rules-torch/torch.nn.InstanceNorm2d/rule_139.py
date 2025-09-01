import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# feature_size can't exceed product of h and w when 4D, and combined with min shape value (Rule 139)

rule_139 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] < 10000, (If(v["arg2_ndim"] == 4, v["arg1_value"] < Select(v["arg2_shape"], 2) * Select(v["arg2_shape"], 3), True))), (If(v["arg2_ndim"] == 4, Select(v["arg2_range"], 0) > -100, True)))) if n else
          And(And(v["arg1_value"] < 10000, (If(v["arg2_ndim"] == 4, v["arg1_value"] < Select(v["arg2_shape"], 2) * Select(v["arg2_shape"], 3), True))), (If(v["arg2_ndim"] == 4, Select(v["arg2_range"], 0) > -100, True))))
)

def rule_139_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 139
        rule_139(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_139(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

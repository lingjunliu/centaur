import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tensor value_range: shape, dtype match, and valid interval (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 10), v["arg2_dtype"] == v["arg1_dtype"]), v["arg2_ndim"] == 1), Select(v["arg2_shape"], 0) == 2), Select(v["arg2_range"], 0) < Select(v["arg2_range"], 1))) if n else
          And(And(And(And(And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 10), v["arg2_dtype"] == v["arg1_dtype"]), v["arg2_ndim"] == 1), Select(v["arg2_shape"], 0) == 2), Select(v["arg2_range"], 0) < Select(v["arg2_range"], 1)))
)

def rule_11_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 11
        rule_11(solver, {'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)

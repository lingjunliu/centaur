import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The ULTIMATE Complete Rule (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg2_ndim"] == 1, Select(v["arg2_shape"], 0) == 2), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), (Or((And(And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0)), (And(And(And(And(v["arg1_ndim"] == 4, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0), Select(v["arg1_shape"], 3) > 0))))), v["arg1_ndim"] >= 3)) if n else
          And(And(And(And(v["arg2_ndim"] == 1, Select(v["arg2_shape"], 0) == 2), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), (Or((And(And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0)), (And(And(And(And(v["arg1_ndim"] == 4, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0), Select(v["arg1_shape"], 3) > 0))))), v["arg1_ndim"] >= 3))
)

def rule_42_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 42
        rule_42(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Fully validated inputs: audio and sample_rate (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(v["arg1_dtype"] == 7, v["arg1_ndim"] == 2), Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), v["arg2_dtype"] == 3), v["arg2_ndim"] == 0), 10 < Select(v["arg2_range"], 0)), Select(v["arg2_range"], 1) < 500000), -2147483648 <= Select(v["arg2_range"], 0)), Select(v["arg2_range"], 1) <= 2147483647)) if n else
          And(And(And(And(And(And(And(And(And(v["arg1_dtype"] == 7, v["arg1_ndim"] == 2), Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), v["arg2_dtype"] == 3), v["arg2_ndim"] == 0), 10 < Select(v["arg2_range"], 0)), Select(v["arg2_range"], 1) < 500000), -2147483648 <= Select(v["arg2_range"], 0)), Select(v["arg2_range"], 1) <= 2147483647))
)

def rule_44_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 44
        rule_44(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)

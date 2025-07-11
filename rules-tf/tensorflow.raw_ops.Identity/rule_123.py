import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If a 2D tensor's dimensions match, the sum must be positive if of float data type (Rule 123)

rule_123 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_dtype"] > 5, v["arg1_dtype"] < 9), v["arg1_ndim"] == 2), Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], 1)), Select(v["arg1_range"], 0) + Select(v["arg1_range"], 1) > 0, False)) if n else
          If(And(And(And(v["arg1_dtype"] > 5, v["arg1_dtype"] < 9), v["arg1_ndim"] == 2), Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], 1)), Select(v["arg1_range"], 0) + Select(v["arg1_range"], 1) > 0, False))
)

def rule_123_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 123
        rule_123(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_123(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)

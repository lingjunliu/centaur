import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If one dimension of the input shape is zero then the number of dimensions of new shape must also be zero, or input_indices dimensions must be zero (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(Or((Or([And(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == 0, v["arg2_ndim"] == 0)) for i in range(6)])), Select(v["arg3_range"], 0) == 0)) if n else
          Or((Or([And(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) == 0, v["arg2_ndim"] == 0)) for i in range(6)])), Select(v["arg3_range"], 0) == 0))
)

def rule_98_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 98
        rule_98(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_range': arg3['range']}, neg)

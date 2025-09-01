import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Shape of paddings is valid with respect to the input tensor's ndim (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or((And(And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 0), Select(v["arg2_shape"], 1) == 2)), (And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 1), Select(v["arg2_shape"], 1) == 2))), (And(And(And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 2), Select(v["arg2_shape"], 1) == 2))), (And(And(And(v["arg1_ndim"] == 3, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 3), Select(v["arg2_shape"], 1) == 2))), (And(And(And(v["arg1_ndim"] == 4, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 4), Select(v["arg2_shape"], 1) == 2)))) if n else
          Or(Or(Or(Or((And(And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 0), Select(v["arg2_shape"], 1) == 2)), (And(And(And(v["arg1_ndim"] == 1, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 1), Select(v["arg2_shape"], 1) == 2))), (And(And(And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 2), Select(v["arg2_shape"], 1) == 2))), (And(And(And(v["arg1_ndim"] == 3, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 3), Select(v["arg2_shape"], 1) == 2))), (And(And(And(v["arg1_ndim"] == 4, v["arg2_ndim"] == 2), Select(v["arg2_shape"], 0) == 4), Select(v["arg2_shape"], 1) == 2))))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 46
        rule_46(solver, {'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)

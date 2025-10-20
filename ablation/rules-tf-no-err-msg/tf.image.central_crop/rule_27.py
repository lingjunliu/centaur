import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the image is 3D or 4D, the central_fraction should result to non-zero height and width. (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(Or((And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * v["arg2_value"] > 0), Select(v["arg1_shape"], 1) * v["arg2_value"] > 0)), (And(And(v["arg1_ndim"] == 4, Select(v["arg1_shape"], 1) * v["arg2_value"] > 0), Select(v["arg1_shape"], 2) * v["arg2_value"] > 0)))) if n else
          Or((And(And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * v["arg2_value"] > 0), Select(v["arg1_shape"], 1) * v["arg2_value"] > 0)), (And(And(v["arg1_ndim"] == 4, Select(v["arg1_shape"], 1) * v["arg2_value"] > 0), Select(v["arg1_shape"], 2) * v["arg2_value"] > 0))))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)

        # Constraints for rule 27
        rule_27(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

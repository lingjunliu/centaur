import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input tensor's last two dimensions are one, num_lower and num_upper should both be less than one (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1), Select(v["arg1_shape"], v["arg1_ndim"] - 2) == 1), And(Select(v["arg2_range"], 0) < 1, Select(v["arg3_range"], 0) < 1), True)) if n else
          If(And(And(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 1), Select(v["arg1_shape"], v["arg1_ndim"] - 2) == 1), And(Select(v["arg2_range"], 0) < 1, Select(v["arg3_range"], 0) < 1), True))
)

def rule_36_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 36
        rule_36(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg3_range': arg3['range']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# lr, rho, epsilon and momentum must be scalar and non-negative and their shapes must equal 1 (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0), v["arg3_ndim"] == 0), v["arg4_ndim"] == 0), Select(v["arg1_shape"], 0) == 1), Select(v["arg2_shape"], 0) == 1), Select(v["arg3_shape"], 0) == 1), Select(v["arg4_shape"], 0) == 1), Select(v["arg1_range"], 0) >= 0), Select(v["arg2_range"], 0) >= 0), Select(v["arg3_range"], 0) >= 0), Select(v["arg4_range"], 0) >= 0)) if n else
          And(And(And(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0), v["arg3_ndim"] == 0), v["arg4_ndim"] == 0), Select(v["arg1_shape"], 0) == 1), Select(v["arg2_shape"], 0) == 1), Select(v["arg3_shape"], 0) == 1), Select(v["arg4_shape"], 0) == 1), Select(v["arg1_range"], 0) >= 0), Select(v["arg2_range"], 0) >= 0), Select(v["arg3_range"], 0) >= 0), Select(v["arg4_range"], 0) >= 0))
)

def rule_66_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_range = Array('arg4_range', IntSort(), IntSort())

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
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 66
        rule_66(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_range': arg3_range, 'arg3_ndim': arg3_ndim, 'arg4_shape': arg4_shape, 'arg4_range': arg4_range, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_range': arg3['range'], 'arg3_ndim': arg3['ndim'], 'arg4_shape': arg4['shape'], 'arg4_range': arg4['range'], 'arg4_ndim': arg4['ndim']}, neg)

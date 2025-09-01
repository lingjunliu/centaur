import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# lr, l1, l2, l2_shrinkage and lr_power should be non-negative, with default zero handling (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, True, And(Select(v["arg1_range"], 0) >= 0, If(v["arg2_ndim"] == 0, True, And(Select(v["arg2_range"], 0) >= 0, If(v["arg3_ndim"] == 0, True, And(Select(v["arg3_range"], 0) >= 0, If(v["arg4_ndim"] == 0, True, And(Select(v["arg4_range"], 0) >= 0, If(v["arg5_ndim"] == 0, True, Select(v["arg5_range"], 0) >= 0)))))))))) if n else
          If(v["arg1_ndim"] == 0, True, And(Select(v["arg1_range"], 0) >= 0, If(v["arg2_ndim"] == 0, True, And(Select(v["arg2_range"], 0) >= 0, If(v["arg3_ndim"] == 0, True, And(Select(v["arg3_range"], 0) >= 0, If(v["arg4_ndim"] == 0, True, And(Select(v["arg4_range"], 0) >= 0, If(v["arg5_ndim"] == 0, True, Select(v["arg5_range"], 0) >= 0))))))))))
)

def rule_37_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

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
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg4_range = Array('arg4_range', IntSort(), IntSort())
        arg5_ndim = Int('arg5_ndim')
        arg5_range = Array('arg5_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_ndim == arg3.ndim)
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_ndim == arg4.ndim)
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))
        solver.add(arg5_ndim == arg5.ndim)
        arg5_range = Store(arg5_range, 0, int(np.min(arg5)))
        arg5_range = Store(arg5_range, 1, int(np.max(arg5)))

        # Constraints for rule 37
        rule_37(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg3_range': arg3_range, 'arg3_ndim': arg3_ndim, 'arg4_range': arg4_range, 'arg4_ndim': arg4_ndim, 'arg5_range': arg5_range, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg3_range': arg3['range'], 'arg3_ndim': arg3['ndim'], 'arg4_range': arg4['range'], 'arg4_ndim': arg4['ndim'], 'arg5_range': arg5['range'], 'arg5_ndim': arg5['ndim']}, neg)

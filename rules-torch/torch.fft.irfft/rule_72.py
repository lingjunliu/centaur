import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if n is given then out should have the same dim size and within valid size  (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg1_value"] == 0), (And(And(And(v["arg2_value"] >= (0 - v["arg3_ndim"]), v["arg2_value"] < v["arg3_ndim"]), (v["arg1_value"] == Select(v["arg3_shape"], v["arg2_value"]))), (v["arg1_value"] >= 2 * (Select(v["arg4_shape"], v["arg2_value"]) - 1)))))) if n else
          Or((v["arg1_value"] == 0), (And(And(And(v["arg2_value"] >= (0 - v["arg3_ndim"]), v["arg2_value"] < v["arg3_ndim"]), (v["arg1_value"] == Select(v["arg3_shape"], v["arg2_value"]))), (v["arg1_value"] >= 2 * (Select(v["arg4_shape"], v["arg2_value"]) - 1))))))
)

def rule_72_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 72
        rule_72(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape'], 'arg4_shape': arg4['shape']}, neg)

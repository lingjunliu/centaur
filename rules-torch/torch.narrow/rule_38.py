import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# if start is a tensor and the dimension size is zero, then length = 0 or start is out of bounds (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], If(v["arg2_value"] < 0, v["arg2_value"] + v["arg1_ndim"], v["arg2_value"])) == 0, Or(v["arg4_value"] == 0, Select(v["arg3_range"], 1) >= Select(v["arg1_shape"], If(v["arg2_value"] < 0, v["arg2_value"] + v["arg1_ndim"], v["arg2_value"]))), False)) if n else
          If(Select(v["arg1_shape"], If(v["arg2_value"] < 0, v["arg2_value"] + v["arg1_ndim"], v["arg2_value"])) == 0, Or(v["arg4_value"] == 0, Select(v["arg3_range"], 1) >= Select(v["arg1_shape"], If(v["arg2_value"] < 0, v["arg2_value"] + v["arg1_ndim"], v["arg2_value"]))), False))
)

def rule_38_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 38
        rule_38(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_range': arg3_range, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range'], 'arg4_value': arg4['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If count_include_pad is False, then if padding is non-zero, then input dimensions must be larger than kernel size + padding * 2  (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, v["arg2_value"] > 0), And(And(v["arg4_ndim"] == 4, Select(v["arg4_shape"], 2) > v["arg3_value"] + v["arg2_value"] * 2), Select(v["arg4_shape"], 3) > v["arg3_value"] + v["arg2_value"] * 2), True)) if n else
          If(And(v["arg1_value"] == False, v["arg2_value"] > 0), And(And(v["arg4_ndim"] == 4, Select(v["arg4_shape"], 2) > v["arg3_value"] + v["arg2_value"] * 2), Select(v["arg4_shape"], 3) > v["arg3_value"] + v["arg2_value"] * 2), True))
)

def rule_66_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 66
        rule_66(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim'], 'arg4_shape': arg4['shape']}, neg)

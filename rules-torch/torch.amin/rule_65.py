import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if out tensor is provided, its shape must be correct when dim is an int. (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == True, And(v["arg1_ndim"] == v["arg4_ndim"], (And([Implies(i < (v["arg1_ndim"] - 1 + 1), If(i == v["arg2_value"], Select(v["arg4_shape"], i) == 1, Select(v["arg4_shape"], i) == Select(v["arg1_shape"], i))) for i in range(6)]))), v["arg4_ndim"] == v["arg1_ndim"] - 1)) if n else
          If(v["arg3_value"] == True, And(v["arg1_ndim"] == v["arg4_ndim"], (And([Implies(i < (v["arg1_ndim"] - 1 + 1), If(i == v["arg2_value"], Select(v["arg4_shape"], i) == 1, Select(v["arg4_shape"], i) == Select(v["arg1_shape"], i))) for i in range(6)]))), v["arg4_ndim"] == v["arg1_ndim"] - 1))
)

def rule_65_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == arg3)
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 65
        rule_65(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim'], 'arg4_shape': arg4['shape']}, neg)

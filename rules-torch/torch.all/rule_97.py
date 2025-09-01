import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if keepdim is True and dim is specified and out is specified, then if the dimension specified is valid, then its shape must be equal to 1, else it's original dimension size (Rule 97)

rule_97 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] == True, If(And((0 - v["arg1_ndim"]) <= v["arg3_value"], v["arg3_value"] < v["arg1_ndim"]), Select(v["arg2_shape"], v["arg3_value"]) == 1, True), True)) if n else
          If(v["arg4_value"] == True, If(And((0 - v["arg1_ndim"]) <= v["arg3_value"], v["arg3_value"] < v["arg1_ndim"]), Select(v["arg2_shape"], v["arg3_value"]) == 1, True), True))
)

def rule_97_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 97
        rule_97(solver, {'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_97(solver, {'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

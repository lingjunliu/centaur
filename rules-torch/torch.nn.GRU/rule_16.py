import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If batch_first is false, the last dimension of input must match input_size. Otherwise it must match hidden_size if h_0 exists (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], Select(v["arg2_shape"], v["arg2_ndim"] - 1) == v["arg3_value"], Select(v["arg4_shape"], v["arg4_ndim"] - 1) == v["arg5_value"])) if n else
          If(v["arg1_value"], Select(v["arg2_shape"], v["arg2_ndim"] - 1) == v["arg3_value"], Select(v["arg4_shape"], v["arg4_ndim"] - 1) == v["arg5_value"]))
)

def rule_16_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg5_value == int(arg5))

        # Constraints for rule 16
        rule_16(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim, 'arg4_shape': arg4_shape, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim'], 'arg4_shape': arg4['shape'], 'arg5_value': arg5['value']}, neg)

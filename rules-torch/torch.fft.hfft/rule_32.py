import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If n is provided, it dictates the output tensor's last dimension, input and output have same number of dimensions (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not((And(Select(v["arg2_shape"], v["arg2_ndim"] - 1) == v["arg1_value"], v["arg2_ndim"] == v["arg3_ndim"]))) if n else
          (And(Select(v["arg2_shape"], v["arg2_ndim"] - 1) == v["arg1_value"], v["arg2_ndim"] == v["arg3_ndim"])))
)

def rule_32_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 32
        rule_32(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_ndim': arg3['ndim']}, neg)

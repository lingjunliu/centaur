import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input is double or float, the output tensor, if present, must have the same dtype AND the size has to match the bins size (Rule 135)

rule_135 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), If(v["arg2_ndim"] > 0, And(v["arg2_dtype"] == v["arg1_dtype"], Select(v["arg2_shape"], 0) == v["arg3_value"]), True), True)) if n else
          If(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), If(v["arg2_ndim"] > 0, And(v["arg2_dtype"] == v["arg1_dtype"], Select(v["arg2_shape"], 0) == v["arg3_value"]), True), True))
)

def rule_135_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 135
        rule_135(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_135(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

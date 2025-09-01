import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# out tensor must have the same length as bins and data type as input, if out is given and the input is not boolean or integer (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] > 0, If(v["arg3_ndim"] > 0, And(Select(v["arg3_shape"], 0) == v["arg1_value"], (If(And(v["arg2_dtype"] > 0, v["arg2_dtype"] < 6), True, v["arg3_dtype"] == v["arg2_dtype"]))), True))) if n else
          And(v["arg1_value"] > 0, If(v["arg3_ndim"] > 0, And(Select(v["arg3_shape"], 0) == v["arg1_value"], (If(And(v["arg2_dtype"] > 0, v["arg2_dtype"] < 6), True, v["arg3_dtype"] == v["arg2_dtype"]))), True)))
)

def rule_50_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 50
        rule_50(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg3_shape': arg3_shape, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg3_shape': arg3['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim']}, neg)

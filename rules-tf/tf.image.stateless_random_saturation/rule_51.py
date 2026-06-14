import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# image last dim is 3 AND lower >=0 AND upper > lower AND seed shape (2, (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3, v["arg2_value"] >= 0), v["arg3_value"] > v["arg2_value"]), v["arg4_ndim"] == 1), Select(v["arg4_shape"], 0) == 2), (Or(v["arg4_dtype"] == 3, v["arg4_dtype"] == 4)))) if n else
          And(And(And(And(And(Select(v["arg1_shape"], v["arg1_ndim"] - 1) == 3, v["arg2_value"] >= 0), v["arg3_value"] > v["arg2_value"]), v["arg4_ndim"] == 1), Select(v["arg4_shape"], 0) == 2), (Or(v["arg4_dtype"] == 3, v["arg4_dtype"] == 4))))
)

def rule_51_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 51
        rule_51(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype, 'arg4_ndim': arg4_ndim, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype'], 'arg4_ndim': arg4['ndim'], 'arg4_shape': arg4['shape']}, neg)

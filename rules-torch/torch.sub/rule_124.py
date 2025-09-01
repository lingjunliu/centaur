import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if other is not a tensor, then other has to be small, check against max value of float16 to avoid overflow, check when other is float and alpha is not too small or not too large and v1 is non-empty, alpha must be not too large and other dim0 must be valid and shape of the input must be non zero in its dimensions and input maximum must be small  (Rule 124)

rule_124 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_dtype"] < 6, And(And(And(And(And(And(v["arg1_value"] < 65500, v["arg1_value"] > -65500), v["arg2_value"] < 1000), v["arg2_value"] > -1000), Select(v["arg3_range"], 0) < 10000), Select(v["arg3_shape"], 0) > 0), And([Implies(i < (v["arg3_ndim"] - 1 + 1), And(Select(v["arg3_shape"], i) > 0, Select(v["arg3_range"], 1) < 65500)) for i in range(6)])), And(v["arg2_value"] < 65500, v["arg2_value"] > -65500))) if n else
          If(v["arg3_dtype"] < 6, And(And(And(And(And(And(v["arg1_value"] < 65500, v["arg1_value"] > -65500), v["arg2_value"] < 1000), v["arg2_value"] > -1000), Select(v["arg3_range"], 0) < 10000), Select(v["arg3_shape"], 0) > 0), And([Implies(i < (v["arg3_ndim"] - 1 + 1), And(Select(v["arg3_shape"], i) > 0, Select(v["arg3_range"], 1) < 65500)) for i in range(6)])), And(v["arg2_value"] < 65500, v["arg2_value"] > -65500)))
)

def rule_124_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 124
        rule_124(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_range': arg3_range, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_124(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)

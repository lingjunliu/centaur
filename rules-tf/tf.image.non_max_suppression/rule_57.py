import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# boxes and scores dimensions, type, matching first dimension, iou_threshold within range, and max_output_size is positive (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), (Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 6)))), (Or((v["arg2_dtype"] == 7), (v["arg2_dtype"] == 6)))), v["arg3_value"] >= 0), v["arg3_value"] <= 1), v["arg4_value"] > 0), v["arg5_value"] >= 0)) if n else
          And(And(And(And(And(And(And(And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), (Or((v["arg1_dtype"] == 7), (v["arg1_dtype"] == 6)))), (Or((v["arg2_dtype"] == 7), (v["arg2_dtype"] == 6)))), v["arg3_value"] >= 0), v["arg3_value"] <= 1), v["arg4_value"] > 0), v["arg5_value"] >= 0))
)

def rule_57_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False
        if not isinstance(arg5, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Real('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Real('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == arg5)

        # Constraints for rule 57
        rule_57(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtype is uint8 and channel = 1, the output tensor must have values between 0 and 255, and shape[-1] must be 1 if rank >= 2 (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 5, v["arg2_value"] == 1), And(And(Select(v["arg3_range"], 0) >= 0, Select(v["arg3_range"], 1) <= 255), (If(v["arg3_ndim"] >= 2, Select(v["arg3_shape"], v["arg3_ndim"] - 1) == 1, True))), True)) if n else
          If(And(v["arg1_value"] == 5, v["arg2_value"] == 1), And(And(Select(v["arg3_range"], 0) >= 0, Select(v["arg3_range"], 1) <= 255), (If(v["arg3_ndim"] >= 2, Select(v["arg3_shape"], v["arg3_ndim"] - 1) == 1, True))), True))
)

def rule_46_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 46
        rule_46(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_range': arg3_range, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_range': arg3['range'], 'arg3_ndim': arg3['ndim']}, neg)

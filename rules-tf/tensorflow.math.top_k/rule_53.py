import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If certain conditions are true regarding input tensor's dtype and also no index_type specified, ensure k is less than certain thresholds and dimension size (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] == 12, (Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 5))), And(And(v["arg2_value"] < 2147483648, (If(v["arg1_dtype"] == 1, v["arg2_value"] < 128, If(v["arg1_dtype"] == 2, v["arg2_value"] < 32768, v["arg2_value"] < 256)))), v["arg2_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1)), If(v["arg3_value"] == 12, And(v["arg2_value"] < 2147483648, v["arg2_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1)), True))) if n else
          If(And(v["arg3_value"] == 12, (Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 5))), And(And(v["arg2_value"] < 2147483648, (If(v["arg1_dtype"] == 1, v["arg2_value"] < 128, If(v["arg1_dtype"] == 2, v["arg2_value"] < 32768, v["arg2_value"] < 256)))), v["arg2_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1)), If(v["arg3_value"] == 12, And(v["arg2_value"] < 2147483648, v["arg2_value"] <= Select(v["arg1_shape"], v["arg1_ndim"] - 1)), True)))
)

def rule_53_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 53
        rule_53(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

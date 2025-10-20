import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If a dtype is specified, the input tensor must be a valid type, and the output tensor, if any, should also have that type. Also, dim should be within the valid range. (Rule 84)

rule_84 = lambda s, v, n=False: (
    s.add(Not(And((And(v["arg2_value"] >= (0 - v["arg1_ndim"]), v["arg2_value"] < v["arg1_ndim"])), (If(v["arg4_value"] != 12, (And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11), v["arg3_dtype"] == v["arg4_value"])), (And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11)))))) if n else
          And((And(v["arg2_value"] >= (0 - v["arg1_ndim"]), v["arg2_value"] < v["arg1_ndim"])), (If(v["arg4_value"] != 12, (And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11), v["arg3_dtype"] == v["arg4_value"])), (And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11))))))
)

def rule_84_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 84
        rule_84(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_84(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value']}, neg)

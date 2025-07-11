import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtype is specified, mean, stddev, shape and output tensor must be of compatible dtype (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] != 0, And(And(And((v["arg1_dtype"] == v["arg3_value"]), (v["arg2_dtype"] == v["arg3_value"])), (v["arg4_dtype"] == v["arg3_value"])), (Or(Or(Or(v["arg5_dtype"] == 2, v["arg5_dtype"] == 3), v["arg5_dtype"] == 4), v["arg5_dtype"] == 5))), False)) if n else
          If(v["arg3_value"] != 0, And(And(And((v["arg1_dtype"] == v["arg3_value"]), (v["arg2_dtype"] == v["arg3_value"])), (v["arg4_dtype"] == v["arg3_value"])), (Or(Or(Or(v["arg5_dtype"] == 2, v["arg5_dtype"] == 3), v["arg5_dtype"] == 4), v["arg5_dtype"] == 5))), False))
)

def rule_34_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')
        arg4_dtype = Int('arg4_dtype')
        arg5_dtype = Int('arg5_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))

        # Constraints for rule 34
        rule_34(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype, 'arg5_dtype': arg5_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype'], 'arg5_dtype': arg5['dtype']}, neg)

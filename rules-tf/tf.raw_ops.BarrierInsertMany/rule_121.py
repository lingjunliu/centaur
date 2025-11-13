import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Shape of values should be divisible by 2, if the component_index is also divisible by 2 and shape of values is not zero and keys shape should also be even and dtype of handle must be equal to string and its shape must be more than 10 and max of values must be smaller than max of handle and the dtype of values should be integer (Rule 121)

rule_121 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg2_value"] % 2 == 0), (Select(v["arg1_shape"], 0) != 0)), And(And(And(And(And(Select(v["arg1_shape"], 0) % 2 == 0, Select(v["arg3_shape"], 0) % 2 == 0), v["arg4_dtype"] == 11), Select(v["arg4_shape"], 0) > 10), Select(v["arg1_range"], 1) < Select(v["arg4_range"], 1)), v["arg1_dtype"] < 6), True)) if n else
          If(And((v["arg2_value"] % 2 == 0), (Select(v["arg1_shape"], 0) != 0)), And(And(And(And(And(Select(v["arg1_shape"], 0) % 2 == 0, Select(v["arg3_shape"], 0) % 2 == 0), v["arg4_dtype"] == 11), Select(v["arg4_shape"], 0) > 10), Select(v["arg1_range"], 1) < Select(v["arg4_range"], 1)), v["arg1_dtype"] < 6), True))
)

def rule_121_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')
        arg4_range = Array('arg4_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))

        # Constraints for rule 121
        rule_121(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg4_dtype': arg4_dtype, 'arg4_shape': arg4_shape, 'arg4_range': arg4_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg4_dtype': arg4['dtype'], 'arg4_shape': arg4['shape'], 'arg4_range': arg4['range']}, neg)

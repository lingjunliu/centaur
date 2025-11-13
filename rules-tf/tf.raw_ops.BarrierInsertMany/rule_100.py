import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Shape of values should be divisible by 2, if the component_index is also divisible by 2 and shape of values is not zero and keys shape should also be even and dtype of handle must be equal to string (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg2_value"] % 2 == 0), (Select(v["arg1_shape"], 0) != 0)), And(And(Select(v["arg1_shape"], 0) % 2 == 0, Select(v["arg3_shape"], 0) % 2 == 0), v["arg4_dtype"] == 11), True)) if n else
          If(And((v["arg2_value"] % 2 == 0), (Select(v["arg1_shape"], 0) != 0)), And(And(Select(v["arg1_shape"], 0) % 2 == 0, Select(v["arg3_shape"], 0) % 2 == 0), v["arg4_dtype"] == 11), True))
)

def rule_100_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 100
        rule_100(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg4_dtype': arg4['dtype']}, neg)

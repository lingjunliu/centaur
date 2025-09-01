import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the keys is string, then the max value of component index should be less than 1024 and it should be greater than -1024 and values should have length greater than component_index (Rule 92)

rule_92 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, And(And(v["arg2_value"] < 1024, v["arg2_value"] > -1024), Select(v["arg3_shape"], 0) > v["arg2_value"]), True)) if n else
          If(v["arg1_dtype"] == 11, And(And(v["arg2_value"] < 1024, v["arg2_value"] > -1024), Select(v["arg3_shape"], 0) > v["arg2_value"]), True))
)

def rule_92_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 92
        rule_92(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_92(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)

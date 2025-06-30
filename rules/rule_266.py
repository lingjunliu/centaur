import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If tensor data type is not bool and if ...ij->...ji string has been chosen then the min value should be smaller or equals to the shape[0] on axis 0. (Rule 266)

rule_266 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_dtype"] != 0), (v["arg2_value"] == 4)), Select(v["arg1_range"], 0) <= Select(v["arg1_shape"], 0), False)) if n else
          If(And((v["arg1_dtype"] != 0), (v["arg2_value"] == 4)), Select(v["arg1_range"], 0) <= Select(v["arg1_shape"], 0), False))
)

def rule_266_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 266
        rule_266(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_266(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)

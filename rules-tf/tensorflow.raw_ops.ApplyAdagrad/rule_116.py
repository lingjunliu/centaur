import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if update_slots is true, and lr has float dtype, then all lr's value needs to be greater than 0, shape needs to be greater or equal to 1 (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == True, (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), And(Select(v["arg2_range"], 0) > 0, Select(v["arg2_shape"], 0) >= 1), False)) if n else
          If(And(v["arg1_value"] == True, (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), And(Select(v["arg2_range"], 0) > 0, Select(v["arg2_shape"], 0) >= 1), False))
)

def rule_116_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 116
        rule_116(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range'], 'arg2_shape': arg2['shape']}, neg)

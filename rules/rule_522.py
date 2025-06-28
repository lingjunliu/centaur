import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If string v_1 is 'tanh' and bool v_2 is True, then shape of dimension 0 for tensor v_3 must be greater than its maximum value (Rule 522)

rule_522 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 11, v["arg2_value"]), Select(v["arg3_shape"], 0) > Select(v["arg3_range"], 1), False)) if n else
          If(And(v["arg1_value"] == 11, v["arg2_value"]), Select(v["arg3_shape"], 0) > Select(v["arg3_range"], 1), False))
)

def rule_522_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == arg2)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 522
        rule_522(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_range': arg3_range, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_522(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range'], 'arg3_shape': arg3['shape']}, neg)

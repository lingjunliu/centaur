import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if float v_1 is positive, then bool v_2 must be true if maximum value in tensor v_3 is equal to the shape of v_3 in the dimension 0 (Rule 425)

rule_425 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 0, If(Select(v["arg3_range"], 1) == Select(v["arg3_shape"], 0), v["arg2_value"], False), False)) if n else
          If(v["arg1_value"] > 0, If(Select(v["arg3_range"], 1) == Select(v["arg3_shape"], 0), v["arg2_value"], False), False))
)

def rule_425_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 425
        rule_425(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_range': arg3_range, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_425(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range'], 'arg3_shape': arg3['shape']}, neg)

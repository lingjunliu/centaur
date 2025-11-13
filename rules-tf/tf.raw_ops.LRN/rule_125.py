import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If image is smaller than 16x16 then alpha should be less than 2 and beta smaller than 0.5 and depth radius smaller than 3, AND channels >2 (Rule 125)

rule_125 = lambda s, v, n=False: (
    s.add(Not(If(And(And(Select(v["arg1_shape"], 2) < 16, Select(v["arg1_shape"], 3) < 16), Select(v["arg1_shape"], 3) > 2), And(And(v["arg2_value"] < 2, v["arg3_value"] < 0.5), v["arg4_value"] < 3), True)) if n else
          If(And(And(Select(v["arg1_shape"], 2) < 16, Select(v["arg1_shape"], 3) < 16), Select(v["arg1_shape"], 3) > 2), And(And(v["arg2_value"] < 2, v["arg3_value"] < 0.5), v["arg4_value"] < 3), True))
)

def rule_125_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 125
        rule_125(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_125(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

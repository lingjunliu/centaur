import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If ceil_mode, padding and other param are under shape and are within valid max range (Rule 109)

rule_109 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And(And(And(And(And(And(Select(v["arg2_shape"], 2) > v["arg3_value"], Select(v["arg2_shape"], 3) > v["arg3_value"]), Select(v["arg2_shape"], 4) > v["arg3_value"]), v["arg3_value"] < 2095166950), (Select(v["arg2_shape"], 2) + 2 * v["arg3_value"] - v["arg4_value"]) < 2147483647), (Select(v["arg2_shape"], 3) + 2 * v["arg3_value"] - v["arg4_value"]) < 2147483647), (Select(v["arg2_shape"], 4) + 2 * v["arg3_value"] - v["arg4_value"]) < 2147483647), False)) if n else
          If(v["arg1_value"] == True, And(And(And(And(And(And(Select(v["arg2_shape"], 2) > v["arg3_value"], Select(v["arg2_shape"], 3) > v["arg3_value"]), Select(v["arg2_shape"], 4) > v["arg3_value"]), v["arg3_value"] < 2095166950), (Select(v["arg2_shape"], 2) + 2 * v["arg3_value"] - v["arg4_value"]) < 2147483647), (Select(v["arg2_shape"], 3) + 2 * v["arg3_value"] - v["arg4_value"]) < 2147483647), (Select(v["arg2_shape"], 4) + 2 * v["arg3_value"] - v["arg4_value"]) < 2147483647), False))
)

def rule_109_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 109
        rule_109(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_109(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

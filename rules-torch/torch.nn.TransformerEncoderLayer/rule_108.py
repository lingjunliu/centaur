import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If activation is not relu or gelu, and bias is false, then the variance of input tensor should be less than 0.01 and the gradient also has to be smaller, d_model must be greater than zero (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] != 12, v["arg1_value"] != 18), v["arg3_value"] == False), And(And(Select(v["arg2_range"], 1) - Select(v["arg2_range"], 0) < 0.01, Select(v["arg2_range"], 1) < 1e-3), v["arg4_value"] > 0), False)) if n else
          If(And(And(v["arg1_value"] != 12, v["arg1_value"] != 18), v["arg3_value"] == False), And(And(Select(v["arg2_range"], 1) - Select(v["arg2_range"], 0) < 0.01, Select(v["arg2_range"], 1) < 1e-3), v["arg4_value"] > 0), False))
)

def rule_108_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 108
        rule_108(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

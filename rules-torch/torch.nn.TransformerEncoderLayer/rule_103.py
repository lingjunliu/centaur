import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If activation is not relu or gelu, and bias is false, then the variance of input tensor should be less than 0.01 and the gradient also has to be smaller (Rule 103)

rule_103 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] != 12, v["arg1_value"] != 18), v["arg3_value"] == False), And(Select(v["arg2_range"], 1) - Select(v["arg2_range"], 0) < 0.01, Select(v["arg2_range"], 1) < 1e-3), False)) if n else
          If(And(And(v["arg1_value"] != 12, v["arg1_value"] != 18), v["arg3_value"] == False), And(Select(v["arg2_range"], 1) - Select(v["arg2_range"], 0) < 0.01, Select(v["arg2_range"], 1) < 1e-3), False))
)

def rule_103_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.torch.index(arg1))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 103
        rule_103(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value']}, neg)

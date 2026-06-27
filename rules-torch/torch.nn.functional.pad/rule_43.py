import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Mode must be one of the supported modes and if replicate, value must be zero, padding must be non-negative (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(Or(Or(v["arg1_value"] == 21, v["arg1_value"] == 22), v["arg1_value"] == 23), v["arg1_value"] == 24)), (If(v["arg1_value"] == 23, v["arg2_value"] == 0, True))), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) >= 0) for i in range(6)]))) if n else
          And(And((Or(Or(Or(v["arg1_value"] == 21, v["arg1_value"] == 22), v["arg1_value"] == 23), v["arg1_value"] == 24)), (If(v["arg1_value"] == 23, v["arg2_value"] == 0, True))), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) >= 0) for i in range(6)])))
)

def rule_43_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 43
        rule_43(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)

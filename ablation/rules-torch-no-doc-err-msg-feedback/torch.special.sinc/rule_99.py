import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if the given string has value relu and tensor dtype is int then the list element should be greater than 0 (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg2_value"] == 11), (And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5))), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)]), True)) if n else
          If(And((v["arg2_value"] == 11), (And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5))), And([Implies(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)]), True))
)

def rule_99_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 99
        rule_99(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_values': arg3_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length']}, neg)

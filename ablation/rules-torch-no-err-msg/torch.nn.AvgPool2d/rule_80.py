import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If count_include_pad is True and kernel_size is specified as a tuple, then each dimension of kernel_size must be positive (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And(And(v["arg2_length"] == 2, Select(v["arg2_values"], 0) > 0), Select(v["arg2_values"], 1) > 0), True)) if n else
          If(v["arg1_value"] == True, And(And(v["arg2_length"] == 2, Select(v["arg2_values"], 0) > 0), Select(v["arg2_values"], 1) > 0), True))
)

def rule_80_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 80
        rule_80(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)

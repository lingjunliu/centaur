import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If count_include_pad is True and kernel_size is tuple, then each dimension of kernel_size multiplied by itself must be less than or equal than divisor_override when divisor_override is not zero. (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == True, v["arg3_value"] > 0), Select(v["arg2_values"], 0) * Select(v["arg2_values"], 1) <= v["arg3_value"], True)) if n else
          If(And(v["arg1_value"] == True, v["arg3_value"] > 0), Select(v["arg2_values"], 0) * Select(v["arg2_values"], 1) <= v["arg3_value"], True))
)

def rule_63_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 63
        rule_63(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value']}, neg)

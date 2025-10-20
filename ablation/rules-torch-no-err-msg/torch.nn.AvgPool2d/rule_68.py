import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If count_include_pad is False, and if padding is a tuple that is not all zero, then the product of kernel dimensions must not be 0. (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, (Or(Select(v["arg2_values"], 0) > 0, Select(v["arg2_values"], 1) > 0))), Select(v["arg3_values"], 0) * Select(v["arg3_values"], 1) > 0, True)) if n else
          If(And(v["arg1_value"] == False, (Or(Select(v["arg2_values"], 0) > 0, Select(v["arg2_values"], 1) > 0))), Select(v["arg3_values"], 0) * Select(v["arg3_values"], 1) > 0, True))
)

def rule_68_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values']}, neg)

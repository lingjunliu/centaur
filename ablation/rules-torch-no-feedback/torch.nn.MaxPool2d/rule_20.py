import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel_size as int, stride as int, pad as tuple, dilation as tuple (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg3_length"] == 2), v["arg4_length"] == 2), And([Implies(i < (1 + 1), And(Select(v["arg3_values"], i) >= 0, And([Implies(i < (1 + 1), Select(v["arg4_values"], i) > 0) for i in range(6)]))) for i in range(6)]))) if n else
          And(And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0), v["arg3_length"] == 2), v["arg4_length"] == 2), And([Implies(i < (1 + 1), And(Select(v["arg3_values"], i) >= 0, And([Implies(i < (1 + 1), Select(v["arg4_values"], i) > 0) for i in range(6)]))) for i in range(6)])))
)

def rule_20_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])

        # Constraints for rule 20
        rule_20(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_length': arg4_length, 'arg4_values': arg4_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_length': arg4['length'], 'arg4_values': arg4['values']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Prevent division by zero in output shape calculation, given kernel, padding, dilation and stride tuples (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_length"] == 3, v["arg2_length"] == 3), v["arg3_length"] == 3), And([Implies(i < (2 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)]))) if n else
          And(And(And(v["arg1_length"] == 3, v["arg2_length"] == 3), v["arg3_length"] == 3), And([Implies(i < (2 + 1), Select(v["arg3_values"], i) > 0) for i in range(6)])))
)

def rule_15_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 15
        rule_15(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)

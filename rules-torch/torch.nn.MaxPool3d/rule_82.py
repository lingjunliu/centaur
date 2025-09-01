import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If kernel_size is a tuple, each element of padding must be a non-negative integer (Rule 82)

rule_82 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg2_length"] == 3, v["arg1_length"] == 3), And([Implies(i < (2 + 1), Select(v["arg2_values"], i) >= 0) for i in range(6)]))) if n else
          And(And(v["arg2_length"] == 3, v["arg1_length"] == 3), And([Implies(i < (2 + 1), Select(v["arg2_values"], i) >= 0) for i in range(6)])))
)

def rule_82_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 82
        rule_82(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_82(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)

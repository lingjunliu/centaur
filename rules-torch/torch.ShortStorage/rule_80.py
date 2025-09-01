import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The product of items in 'size' list (if provided (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(Or([And(p < (1 + 1), And((And([Implies(i < (v["arg1_length"] - 1 + 1), p == p * Select(v["arg1_values"], i)) for i in range(6)])), p < 5000000)) for p in range(6)])) if n else
          Or([And(p < (1 + 1), And((And([Implies(i < (v["arg1_length"] - 1 + 1), p == p * Select(v["arg1_values"], i)) for i in range(6)])), p < 5000000)) for p in range(6)]))
)

def rule_80_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 80
        rule_80(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)

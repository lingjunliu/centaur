import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# indices' elements must be less than the product of shape's elements (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_range"], 1) < (If(v["arg2_length"] == 0, 1, (And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) * Select(v["arg2_values"], i)) for i in range(6)]))))) if n else
          Select(v["arg1_range"], 1) < (If(v["arg2_length"] == 0, 1, (And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) * Select(v["arg2_values"], i)) for i in range(6)])))))
)

def rule_39_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 39
        rule_39(solver, {'arg1_range': arg1_range, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_range': arg1['range'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)

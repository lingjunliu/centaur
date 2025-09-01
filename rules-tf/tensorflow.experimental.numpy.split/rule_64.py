import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If indices_or_sections is a list, it must be strictly increasing and contain only positive elements when the list is not empty and its length is also positive, and each element in list must be less than MAX_INT (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, (And(v["arg1_length"] > 1, And([Implies(i < (v["arg1_length"] - 2 + 1), And(And(Select(v["arg1_values"], i) < Select(v["arg1_values"], i + 1), Select(v["arg1_values"], i) > 0), Select(v["arg1_values"], i) < 2147483647)) for i in range(6)]))), True)) if n else
          If(v["arg1_length"] > 0, (And(v["arg1_length"] > 1, And([Implies(i < (v["arg1_length"] - 2 + 1), And(And(Select(v["arg1_values"], i) < Select(v["arg1_values"], i + 1), Select(v["arg1_values"], i) > 0), Select(v["arg1_values"], i) < 2147483647)) for i in range(6)]))), True))
)

def rule_64_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 64
        rule_64(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)

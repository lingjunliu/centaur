import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# dilations list must have positive elements if it has elements and correct length (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (If(v["arg1_length"] > 0, v["arg1_length"] - 1, 0) + 1), If(v["arg1_length"] > 0, And((Or(Or(v["arg1_length"] == 1, v["arg1_length"] == 2), v["arg1_length"] == 4)), Select(v["arg1_values"], i) > 0), True)) for i in range(6)])) if n else
          And([Implies(i < (If(v["arg1_length"] > 0, v["arg1_length"] - 1, 0) + 1), If(v["arg1_length"] > 0, And((Or(Or(v["arg1_length"] == 1, v["arg1_length"] == 2), v["arg1_length"] == 4)), Select(v["arg1_values"], i) > 0), True)) for i in range(6)]))
)

def rule_102_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 102
        rule_102(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)

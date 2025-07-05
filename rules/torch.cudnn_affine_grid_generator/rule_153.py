import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If output size has enough dimensions and value, check is not smaller than 10 for value of that element. (Rule 153)

rule_153 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] == 4, And(Select(v["arg1_values"], 2) > 10, Select(v["arg1_values"], 3) > 10), If(v["arg1_length"] == 5, And(And(Select(v["arg1_values"], 2) > 10, Select(v["arg1_values"], 3) > 10), Select(v["arg1_values"], 4) > 10), False))) if n else
          If(v["arg1_length"] == 4, And(Select(v["arg1_values"], 2) > 10, Select(v["arg1_values"], 3) > 10), If(v["arg1_length"] == 5, And(And(Select(v["arg1_values"], 2) > 10, Select(v["arg1_values"], 3) > 10), Select(v["arg1_values"], 4) > 10), False)))
)

def rule_153_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 153
        rule_153(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_153(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)

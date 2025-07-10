import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If dilations is a list, and the length is 3, then dilations in the batch and depth dimensions must be 1 (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] == 3, And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 2) == 1), False)) if n else
          If(v["arg1_length"] == 3, And(Select(v["arg1_values"], 0) == 1, Select(v["arg1_values"], 2) == 1), False))
)

def rule_36_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 36
        rule_36(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)

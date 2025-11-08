import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the length of list is greater than 5, and the first element is smaller than the last element, then the first and second elements must have same values. (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] > 5, Select(v["arg1_values"], 0) < Select(v["arg1_values"], v["arg1_length"] - 1)), Select(v["arg1_values"], 0) == Select(v["arg1_values"], 1), True)) if n else
          If(And(v["arg1_length"] > 5, Select(v["arg1_values"], 0) < Select(v["arg1_values"], v["arg1_length"] - 1)), Select(v["arg1_values"], 0) == Select(v["arg1_values"], 1), True))
)

def rule_108_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 108
        rule_108(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)

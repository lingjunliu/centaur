import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the list v_1 has less than 10 elements and element with index 0 is less than or equal to 5, the element with index 1 must be greater or equal to 5 (Rule 126)

rule_126 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] < 10, Select(v["arg1_values"], 0) <= 5), Select(v["arg1_values"], 1) >= 5, True)) if n else
          If(And(v["arg1_length"] < 10, Select(v["arg1_values"], 0) <= 5), Select(v["arg1_values"], 1) >= 5, True))
)

def rule_126_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 126
        rule_126(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_126(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)

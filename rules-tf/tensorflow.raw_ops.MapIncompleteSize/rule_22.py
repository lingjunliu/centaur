import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtypes is not empty then at least one of the dtypes should be an integer or floating point number (1-9 (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, Or([And(i < (v["arg1_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], i) == 1, Select(v["arg1_values"], i) == 2), Select(v["arg1_values"], i) == 3), Select(v["arg1_values"], i) == 4), Select(v["arg1_values"], i) == 5), Select(v["arg1_values"], i) == 6), Select(v["arg1_values"], i) == 7), Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9)) for i in range(6)]), True)) if n else
          If(v["arg1_length"] > 0, Or([And(i < (v["arg1_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], i) == 1, Select(v["arg1_values"], i) == 2), Select(v["arg1_values"], i) == 3), Select(v["arg1_values"], i) == 4), Select(v["arg1_values"], i) == 5), Select(v["arg1_values"], i) == 6), Select(v["arg1_values"], i) == 7), Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9)) for i in range(6)]), True))
)

def rule_22_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 22
        rule_22(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If all the values are of string and each string should contain a valid value for example from list  (Rule 129)

rule_129 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_length"] - 1 + 1), Or(Or(Or(Select(v["arg1_values"], i) == 11, Select(v["arg1_values"], i) == 12), Select(v["arg1_values"], i) == 13), Select(v["arg1_values"], i) == 14)) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_length"] - 1 + 1), Or(Or(Or(Select(v["arg1_values"], i) == 11, Select(v["arg1_values"], i) == 12), Select(v["arg1_values"], i) == 13), Select(v["arg1_values"], i) == 14)) for i in range(6)]))
)

def rule_129_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, str) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), StringSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 129
        rule_129(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_129(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)

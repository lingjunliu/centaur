import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Require all elements of list of strings must be from a certain valid list if scripting is enabled (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And([Implies(i < (v["arg2_length"] - 1 + 1), Or(Or(Or(Select(v["arg2_values"], i) == 20, Select(v["arg2_values"], i) == 11), Select(v["arg2_values"], i) == 12), Select(v["arg2_values"], i) == 13)) for i in range(6)]), True)) if n else
          If(v["arg1_value"] == True, And([Implies(i < (v["arg2_length"] - 1 + 1), Or(Or(Or(Select(v["arg2_values"], i) == 20, Select(v["arg2_values"], i) == 11), Select(v["arg2_values"], i) == 12), Select(v["arg2_values"], i) == 13)) for i in range(6)]), True))
)

def rule_55_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, str) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), StringSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 55
        rule_55(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)

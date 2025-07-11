import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Enforce value properties with flag (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"], (And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) > -5000, Select(v["arg1_values"], i) < 5000), Select(v["arg1_values"], i) >= 0)) for i in range(6)])), False)) if n else
          If(v["arg2_value"], (And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) > -5000, Select(v["arg1_values"], i) < 5000), Select(v["arg1_values"], i) >= 0)) for i in range(6)])), False))
)

def rule_99_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == arg2)

        # Constraints for rule 99
        rule_99(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value']}, neg)

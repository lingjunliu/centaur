import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If a thread's running time is recorded as 'none' then catch exceptions must be true (Rule 146)

rule_146 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) == 6) for i in range(6)]), v["arg2_value"] == True, True)) if n else
          If(Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) == 6) for i in range(6)]), v["arg2_value"] == True, True))
)

def rule_146_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, str) for e in arg1)):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), StringSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == arg2)

        # Constraints for rule 146
        rule_146(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_146(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)

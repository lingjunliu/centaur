import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If List of Bool v_1's length is larger than 0, last element of V_1 must be true (Rule 138)

rule_138 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, Select(v["arg1_values"], v["arg1_length"] - 1) == True, True)) if n else
          If(v["arg1_length"] > 0, Select(v["arg1_values"], v["arg1_length"] - 1) == True, True))
)

def rule_138_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, bool) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), BoolSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 138
        rule_138(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_138(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)

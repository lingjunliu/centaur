import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# ModuleDict values, if they are lists of strings, then either the list must be empty or the strings must be valid. In the latter case we enforce same strings. (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg1_length"] == 0), (And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], 0) == Select(v["arg1_values"], i), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], i) == 0, Select(v["arg1_values"], i) == 1), Select(v["arg1_values"], i) == 2), Select(v["arg1_values"], i) == 3), Select(v["arg1_values"], i) == 4), Select(v["arg1_values"], i) == 5), Select(v["arg1_values"], i) == 6), Select(v["arg1_values"], i) == 7), Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9), Select(v["arg1_values"], i) == 10), Select(v["arg1_values"], i) == 11)))) for i in range(6)])))) if n else
          Or((v["arg1_length"] == 0), (And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], 0) == Select(v["arg1_values"], i), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], i) == 0, Select(v["arg1_values"], i) == 1), Select(v["arg1_values"], i) == 2), Select(v["arg1_values"], i) == 3), Select(v["arg1_values"], i) == 4), Select(v["arg1_values"], i) == 5), Select(v["arg1_values"], i) == 6), Select(v["arg1_values"], i) == 7), Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9), Select(v["arg1_values"], i) == 10), Select(v["arg1_values"], i) == 11)))) for i in range(6)]))))
)

def rule_73_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, str) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), StringSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 73
        rule_73(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)

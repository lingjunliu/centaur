import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# A list of strings should all be unique (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_length"] - 1 + 1), And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) != Select(v["arg1_values"], j)) for j in range(6)])) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_length"] - 1 + 1), And([Implies(j < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) != Select(v["arg1_values"], j)) for j in range(6)])) for i in range(6)]))
)

def rule_29_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 29
        rule_29(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)

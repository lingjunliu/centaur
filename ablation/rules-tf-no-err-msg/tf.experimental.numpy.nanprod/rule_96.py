import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When axis is a tuple of strings each element of the tuple is valid (Rule 96)

rule_96 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg2_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg2_values"], i) == 6, Select(v["arg2_values"], i) == 7), Select(v["arg2_values"], i) == 8), Select(v["arg2_values"], i) == 9), Select(v["arg2_values"], i) == 10), Select(v["arg2_values"], i) == 11), Select(v["arg2_values"], i) == 12), Select(v["arg2_values"], i) == 13), Select(v["arg2_values"], i) == 14), Select(v["arg2_values"], i) == 15), Select(v["arg2_values"], i) == 16), Select(v["arg2_values"], i) == 17), Select(v["arg2_values"], i) == 18), Select(v["arg2_values"], i) == 19), Select(v["arg2_values"], i) == 20), Select(v["arg2_values"], i) == 21), Select(v["arg2_values"], i) == 22), Select(v["arg2_values"], i) == 23), Select(v["arg2_values"], i) == 24), Select(v["arg2_values"], i) == 25)) for i in range(6)])) if n else
          And([Implies(i < (v["arg2_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg2_values"], i) == 6, Select(v["arg2_values"], i) == 7), Select(v["arg2_values"], i) == 8), Select(v["arg2_values"], i) == 9), Select(v["arg2_values"], i) == 10), Select(v["arg2_values"], i) == 11), Select(v["arg2_values"], i) == 12), Select(v["arg2_values"], i) == 13), Select(v["arg2_values"], i) == 14), Select(v["arg2_values"], i) == 15), Select(v["arg2_values"], i) == 16), Select(v["arg2_values"], i) == 17), Select(v["arg2_values"], i) == 18), Select(v["arg2_values"], i) == 19), Select(v["arg2_values"], i) == 20), Select(v["arg2_values"], i) == 21), Select(v["arg2_values"], i) == 22), Select(v["arg2_values"], i) == 23), Select(v["arg2_values"], i) == 24), Select(v["arg2_values"], i) == 25)) for i in range(6)]))
)

def rule_96_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all(isinstance(e, str) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), StringSort())

        # Value assignments
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 96
        rule_96(solver, {'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)

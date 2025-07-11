import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# List of string must be valid, if it is specified (Rule 83)

rule_83 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 0, Or([And(i < (v["arg1_length"] - 1 + 1), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], i) == 0, Select(v["arg1_values"], i) == 1), Select(v["arg1_values"], i) == 2), Select(v["arg1_values"], i) == 3), Select(v["arg1_values"], i) == 4), Select(v["arg1_values"], i) == 5), Select(v["arg1_values"], i) == 6), Select(v["arg1_values"], i) == 7), Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9), Select(v["arg1_values"], i) == 10), Select(v["arg1_values"], i) == 11), Select(v["arg1_values"], i) == 12), Select(v["arg1_values"], i) == 13), Select(v["arg1_values"], i) == 14), Select(v["arg1_values"], i) == 15), Select(v["arg1_values"], i) == 16), Select(v["arg1_values"], i) == 17), Select(v["arg1_values"], i) == 18), Select(v["arg1_values"], i) == 19), Select(v["arg1_values"], i) == 20), Select(v["arg1_values"], i) == 21), Select(v["arg1_values"], i) == 22))) for i in range(6)]), False)) if n else
          If(v["arg1_length"] > 0, Or([And(i < (v["arg1_length"] - 1 + 1), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], i) == 0, Select(v["arg1_values"], i) == 1), Select(v["arg1_values"], i) == 2), Select(v["arg1_values"], i) == 3), Select(v["arg1_values"], i) == 4), Select(v["arg1_values"], i) == 5), Select(v["arg1_values"], i) == 6), Select(v["arg1_values"], i) == 7), Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9), Select(v["arg1_values"], i) == 10), Select(v["arg1_values"], i) == 11), Select(v["arg1_values"], i) == 12), Select(v["arg1_values"], i) == 13), Select(v["arg1_values"], i) == 14), Select(v["arg1_values"], i) == 15), Select(v["arg1_values"], i) == 16), Select(v["arg1_values"], i) == 17), Select(v["arg1_values"], i) == 18), Select(v["arg1_values"], i) == 19), Select(v["arg1_values"], i) == 20), Select(v["arg1_values"], i) == 21), Select(v["arg1_values"], i) == 22))) for i in range(6)]), False))
)

def rule_83_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 83
        rule_83(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_83(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length']}, neg)

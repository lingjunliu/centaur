import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Dummy list of string, first string must be in allowed values (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_length"] > 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], 0) == 0, Select(v["arg1_values"], 0) == 1), Select(v["arg1_values"], 0) == 2), Select(v["arg1_values"], 0) == 3), Select(v["arg1_values"], 0) == 4), Select(v["arg1_values"], 0) == 5), Select(v["arg1_values"], 0) == 6), Select(v["arg1_values"], 0) == 7), Select(v["arg1_values"], 0) == 8), Select(v["arg1_values"], 0) == 9), Select(v["arg1_values"], 0) == 10), Select(v["arg1_values"], 0) == 11), Select(v["arg1_values"], 0) == 12), Select(v["arg1_values"], 0) == 13), Select(v["arg1_values"], 0) == 14), Select(v["arg1_values"], 0) == 15), Select(v["arg1_values"], 0) == 16), Select(v["arg1_values"], 0) == 17), Select(v["arg1_values"], 0) == 18), Select(v["arg1_values"], 0) == 19), Select(v["arg1_values"], 0) == 20), Select(v["arg1_values"], 0) == 21), Select(v["arg1_values"], 0) == 22), Select(v["arg1_values"], 0) == 23), Select(v["arg1_values"], 0) == 24), Select(v["arg1_values"], 0) == 25), Select(v["arg1_values"], 0) == 26), Select(v["arg1_values"], 0) == 27), Select(v["arg1_values"], 0) == 28), Select(v["arg1_values"], 0) == 29), Select(v["arg1_values"], 0) == 20)))) if n else
          And((v["arg1_length"] > 0), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], 0) == 0, Select(v["arg1_values"], 0) == 1), Select(v["arg1_values"], 0) == 2), Select(v["arg1_values"], 0) == 3), Select(v["arg1_values"], 0) == 4), Select(v["arg1_values"], 0) == 5), Select(v["arg1_values"], 0) == 6), Select(v["arg1_values"], 0) == 7), Select(v["arg1_values"], 0) == 8), Select(v["arg1_values"], 0) == 9), Select(v["arg1_values"], 0) == 10), Select(v["arg1_values"], 0) == 11), Select(v["arg1_values"], 0) == 12), Select(v["arg1_values"], 0) == 13), Select(v["arg1_values"], 0) == 14), Select(v["arg1_values"], 0) == 15), Select(v["arg1_values"], 0) == 16), Select(v["arg1_values"], 0) == 17), Select(v["arg1_values"], 0) == 18), Select(v["arg1_values"], 0) == 19), Select(v["arg1_values"], 0) == 20), Select(v["arg1_values"], 0) == 21), Select(v["arg1_values"], 0) == 22), Select(v["arg1_values"], 0) == 23), Select(v["arg1_values"], 0) == 24), Select(v["arg1_values"], 0) == 25), Select(v["arg1_values"], 0) == 26), Select(v["arg1_values"], 0) == 27), Select(v["arg1_values"], 0) == 28), Select(v["arg1_values"], 0) == 29), Select(v["arg1_values"], 0) == 20))))
)

def rule_15_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 15
        rule_15(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)

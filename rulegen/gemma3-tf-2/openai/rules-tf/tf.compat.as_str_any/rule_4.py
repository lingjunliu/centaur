import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tuple of strings contains only supported tokens; encoding is default (Rule 4)

rule_4 = lambda s, v, n=False: (
    s.add(Not(And((And([Implies(i < (v["arg1_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], i) == 0, Select(v["arg1_values"], i) == 1), Select(v["arg1_values"], i) == 2), Select(v["arg1_values"], i) == 3), Select(v["arg1_values"], i) == 4), Select(v["arg1_values"], i) == 5), Select(v["arg1_values"], i) == 6), Select(v["arg1_values"], i) == 7), Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9), Select(v["arg1_values"], i) == 10), Select(v["arg1_values"], i) == 11), Select(v["arg1_values"], i) == 12), Select(v["arg1_values"], i) == 13), Select(v["arg1_values"], i) == 14), Select(v["arg1_values"], i) == 15), Select(v["arg1_values"], i) == 16), Select(v["arg1_values"], i) == 17), Select(v["arg1_values"], i) == 18), Select(v["arg1_values"], i) == 19), Select(v["arg1_values"], i) == 20), Select(v["arg1_values"], i) == 21), Select(v["arg1_values"], i) == 22), Select(v["arg1_values"], i) == 23), Select(v["arg1_values"], i) == 24), Select(v["arg1_values"], i) == 25)) for i in range(6)])), v["arg2_value"] == 6)) if n else
          And((And([Implies(i < (v["arg1_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg1_values"], i) == 0, Select(v["arg1_values"], i) == 1), Select(v["arg1_values"], i) == 2), Select(v["arg1_values"], i) == 3), Select(v["arg1_values"], i) == 4), Select(v["arg1_values"], i) == 5), Select(v["arg1_values"], i) == 6), Select(v["arg1_values"], i) == 7), Select(v["arg1_values"], i) == 8), Select(v["arg1_values"], i) == 9), Select(v["arg1_values"], i) == 10), Select(v["arg1_values"], i) == 11), Select(v["arg1_values"], i) == 12), Select(v["arg1_values"], i) == 13), Select(v["arg1_values"], i) == 14), Select(v["arg1_values"], i) == 15), Select(v["arg1_values"], i) == 16), Select(v["arg1_values"], i) == 17), Select(v["arg1_values"], i) == 18), Select(v["arg1_values"], i) == 19), Select(v["arg1_values"], i) == 20), Select(v["arg1_values"], i) == 21), Select(v["arg1_values"], i) == 22), Select(v["arg1_values"], i) == 23), Select(v["arg1_values"], i) == 24), Select(v["arg1_values"], i) == 25)) for i in range(6)])), v["arg2_value"] == 6))
)

def rule_4_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, str) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), StringSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 4
        rule_4(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_4(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)

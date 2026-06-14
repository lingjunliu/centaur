import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# explicit padding for batch and channel dimensions must be zero (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] == 8, v["arg2_value"] == 24), And(And(And(Select(v["arg1_values"], 0) == 0, Select(v["arg1_values"], 1) == 0), Select(v["arg1_values"], 6) == 0), Select(v["arg1_values"], 7) == 0), If(And(v["arg1_length"] == 8, v["arg2_value"] == 25), And(And(And(Select(v["arg1_values"], 0) == 0, Select(v["arg1_values"], 1) == 0), Select(v["arg1_values"], 2) == 0), Select(v["arg1_values"], 3) == 0), True))) if n else
          If(And(v["arg1_length"] == 8, v["arg2_value"] == 24), And(And(And(Select(v["arg1_values"], 0) == 0, Select(v["arg1_values"], 1) == 0), Select(v["arg1_values"], 6) == 0), Select(v["arg1_values"], 7) == 0), If(And(v["arg1_length"] == 8, v["arg2_value"] == 25), And(And(And(Select(v["arg1_values"], 0) == 0, Select(v["arg1_values"], 1) == 0), Select(v["arg1_values"], 2) == 0), Select(v["arg1_values"], 3) == 0), True)))
)

def rule_19_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 19
        rule_19(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)

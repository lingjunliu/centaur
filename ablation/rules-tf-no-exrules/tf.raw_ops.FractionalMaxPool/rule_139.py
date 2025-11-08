import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the length of pooling_ratio is exactly 5 then the element at index 1 must be greater than the element at index 0 and element at index 3 must be greater than index 2. Furthermore, element at index 0 is greater than zero and element at index 4 is less than 2 and seed must be greater than zero (Rule 139)

rule_139 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] == 5, And(And(And(And(Select(v["arg1_values"], 1) > Select(v["arg1_values"], 0), Select(v["arg1_values"], 3) > Select(v["arg1_values"], 2)), Select(v["arg1_values"], 0) > 0), Select(v["arg1_values"], 4) < 2), v["arg2_value"] > 0), True)) if n else
          If(v["arg1_length"] == 5, And(And(And(And(Select(v["arg1_values"], 1) > Select(v["arg1_values"], 0), Select(v["arg1_values"], 3) > Select(v["arg1_values"], 2)), Select(v["arg1_values"], 0) > 0), Select(v["arg1_values"], 4) < 2), v["arg2_value"] > 0), True))
)

def rule_139_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), RealSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 139
        rule_139(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_139(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)

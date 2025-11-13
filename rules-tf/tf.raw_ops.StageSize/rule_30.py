import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If memory_limit is greater than zero, then the first dtype in dtypes list should be either float16, float32, float64, complex64, complex128 (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] > 0, (Or(Or(Or(Or(Select(v["arg1_values"], 0) == 6, Select(v["arg1_values"], 0) == 7), Select(v["arg1_values"], 0) == 8), Select(v["arg1_values"], 0) == 9), Select(v["arg1_values"], 0) == 10)), True)) if n else
          If(v["arg2_value"] > 0, (Or(Or(Or(Or(Select(v["arg1_values"], 0) == 6, Select(v["arg1_values"], 0) == 7), Select(v["arg1_values"], 0) == 8), Select(v["arg1_values"], 0) == 9), Select(v["arg1_values"], 0) == 10)), True))
)

def rule_30_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 30
        rule_30(solver, {'arg1_values': arg1_values, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_values': arg1['values'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The product of padding values must not exceed a certain threshold (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1) * Select(v["arg2_values"], 0) * Select(v["arg2_values"], 1) <= v["arg3_value"]) if n else
          Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1) * Select(v["arg2_values"], 0) * Select(v["arg2_values"], 1) <= v["arg3_value"])
)

def rule_30_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 30
        rule_30(solver, {'arg1_values': arg1_values, 'arg2_values': arg2_values, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_values': arg1['values'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value']}, neg)

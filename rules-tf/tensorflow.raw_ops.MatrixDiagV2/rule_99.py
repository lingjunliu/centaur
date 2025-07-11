import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If k is a tuple, its values must be less than num_rows or num_cols if they are given (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(And((And(Select(v["arg1_values"], 0) < v["arg2_value"], Or(Select(v["arg1_values"], 1) < v["arg2_value"], v["arg2_value"] == 0))), (And(Select(v["arg1_values"], 0) < v["arg3_value"], Or(Select(v["arg1_values"], 1) < v["arg3_value"], v["arg3_value"] == 0))))) if n else
          And((And(Select(v["arg1_values"], 0) < v["arg2_value"], Or(Select(v["arg1_values"], 1) < v["arg2_value"], v["arg2_value"] == 0))), (And(Select(v["arg1_values"], 0) < v["arg3_value"], Or(Select(v["arg1_values"], 1) < v["arg3_value"], v["arg3_value"] == 0)))))
)

def rule_99_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 99
        rule_99(solver, {'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# year, month, day should represent a valid date (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(Or(Or(Or(Or(Or(v["arg2_value"] == 1, v["arg2_value"] == 3), v["arg2_value"] == 5), v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 10), v["arg2_value"] == 12)), Or(v["arg3_value"] <= 31, (Or(Or(Or(v["arg2_value"] == 4, v["arg2_value"] == 6), v["arg2_value"] == 9), v["arg2_value"] == 11)))), Or(v["arg3_value"] <= 30, v["arg2_value"] == 2)), If(And(v["arg1_value"] % 4 == 0, (Or(v["arg1_value"] % 100 != 0, v["arg1_value"] % 400 == 0))), v["arg3_value"] <= 29, v["arg3_value"] <= 28))) if n else
          And(And(And((Or(Or(Or(Or(Or(Or(v["arg2_value"] == 1, v["arg2_value"] == 3), v["arg2_value"] == 5), v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 10), v["arg2_value"] == 12)), Or(v["arg3_value"] <= 31, (Or(Or(Or(v["arg2_value"] == 4, v["arg2_value"] == 6), v["arg2_value"] == 9), v["arg2_value"] == 11)))), Or(v["arg3_value"] <= 30, v["arg2_value"] == 2)), If(And(v["arg1_value"] % 4 == 0, (Or(v["arg1_value"] % 100 != 0, v["arg1_value"] % 400 == 0))), v["arg3_value"] <= 29, v["arg3_value"] <= 28)))
)

def rule_12_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 12
        rule_12(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

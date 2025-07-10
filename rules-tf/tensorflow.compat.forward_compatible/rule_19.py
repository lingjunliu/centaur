import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The combination of year and month and day should represent a valid date (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(Or([And(m < (12 + 1), And(v["arg2_value"] == m, (If(m == 2, Or([And(d < (If(And(v["arg1_value"] % 4 == 0, Or(v["arg1_value"] % 100 != 0, v["arg1_value"] % 400 == 0)), 29, 28) + 1), v["arg3_value"] == d) for d in range(6)]), If(Or(Or(Or(m == 4, m == 6), m == 9), m == 11), Or([And(d < (30 + 1), v["arg3_value"] == d) for d in range(6)]), Or([And(d < (31 + 1), v["arg3_value"] == d) for d in range(6)])))))) for m in range(6)])) if n else
          Or([And(m < (12 + 1), And(v["arg2_value"] == m, (If(m == 2, Or([And(d < (If(And(v["arg1_value"] % 4 == 0, Or(v["arg1_value"] % 100 != 0, v["arg1_value"] % 400 == 0)), 29, 28) + 1), v["arg3_value"] == d) for d in range(6)]), If(Or(Or(Or(m == 4, m == 6), m == 9), m == 11), Or([And(d < (30 + 1), v["arg3_value"] == d) for d in range(6)]), Or([And(d < (31 + 1), v["arg3_value"] == d) for d in range(6)])))))) for m in range(6)]))
)

def rule_19_func(arg1, arg2, arg3, solver=None, neg=False):
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

        # Constraints for rule 19
        rule_19(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Valid Hz size and make power of 2 for performance gain (Rule 148)

rule_148 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (15 + 1), And(v["arg1_value"] == 2 * i, Or([And(j < (15 + 1), v["arg2_value"] == 2 * j) for j in range(6)]))) for i in range(6)])) if n else
          Or([And(i < (15 + 1), And(v["arg1_value"] == 2 * i, Or([And(j < (15 + 1), v["arg2_value"] == 2 * j) for j in range(6)]))) for i in range(6)]))
)

def rule_148_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 148
        rule_148(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_148(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

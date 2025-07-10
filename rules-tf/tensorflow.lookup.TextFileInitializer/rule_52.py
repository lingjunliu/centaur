import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If key_index is LINE_NUMBER or WHOLE_LINE, the delimiter must be none (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_value"] == -1, v["arg1_value"] == -2), v["arg2_value"] == 6, False)) if n else
          If(Or(v["arg1_value"] == -1, v["arg1_value"] == -2), v["arg2_value"] == 6, False))
)

def rule_52_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 52
        rule_52(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

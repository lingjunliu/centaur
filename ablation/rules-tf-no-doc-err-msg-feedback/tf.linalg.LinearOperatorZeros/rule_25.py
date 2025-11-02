import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If row_dimension or col_dimension are not specified, they default to 1 (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, And(v["arg1_value"] == 1, If(v["arg2_value"] == 0, v["arg2_value"] == 1, True)), True)) if n else
          If(v["arg1_value"] == 0, And(v["arg1_value"] == 1, If(v["arg2_value"] == 0, v["arg2_value"] == 1, True)), True))
)

def rule_25_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 25
        rule_25(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Enforce leap year constraint on the year only. (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(Or((And(v["arg1_value"] % 4 == 0, v["arg1_value"] % 100 != 0)), (v["arg1_value"] % 400 == 0)), True, True)) if n else
          If(Or((And(v["arg1_value"] % 4 == 0, v["arg1_value"] % 100 != 0)), (v["arg1_value"] % 400 == 0)), True, True))
)

def rule_99_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 99
        rule_99(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_value': arg1['value']}, neg)

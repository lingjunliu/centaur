import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if dropout probability is not between 0 and 1, it raises ValueError (Rule 9)

rule_9 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_value"] < 0, v["arg1_value"] > 1), False, False)) if n else
          If(Or(v["arg1_value"] < 0, v["arg1_value"] > 1), False, False))
)

def rule_9_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 9
        rule_9(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_9(solver, {'arg1_value': arg1['value']}, neg)

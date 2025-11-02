import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check the return value from is_queue_empty call (Rule 187)

rule_187 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] == True, v["arg1_value"] == False)) if n else
          Or(v["arg1_value"] == True, v["arg1_value"] == False))
)

def rule_187_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 187
        rule_187(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_187(solver, {'arg1_value': arg1['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# padding must be SAME or VALID (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(v["arg1_value"] == 27, v["arg1_value"] == 26), v["arg1_value"] == 22), v["arg1_value"] == 21)) if n else
          Or(Or(Or(v["arg1_value"] == 27, v["arg1_value"] == 26), v["arg1_value"] == 22), v["arg1_value"] == 21))
)

def rule_80_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))

        # Constraints for rule 80
        rule_80(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_value': arg1['value']}, neg)

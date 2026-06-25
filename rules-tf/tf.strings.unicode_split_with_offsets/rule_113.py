import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# when replacement_char is specified, errors has to be replace, and errors has to be a valid string option (Rule 113)

rule_113 = lambda s, v, n=False: (
    s.add(Not(Or((Or(v["arg1_value"] == 45, v["arg1_value"] == 44)), (And(v["arg2_value"] >= 0, v["arg1_value"] == 43)))) if n else
          Or((Or(v["arg1_value"] == 45, v["arg1_value"] == 44)), (And(v["arg2_value"] >= 0, v["arg1_value"] == 43))))
)

def rule_113_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 113
        rule_113(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_113(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

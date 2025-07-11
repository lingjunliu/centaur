import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The number should be positive if the string function is a linear function, negative if the function is not a linear function. (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 21, v["arg1_value"] > 0, v["arg1_value"] < 0)) if n else
          If(v["arg2_value"] == 21, v["arg1_value"] > 0, v["arg1_value"] < 0))
)

def rule_74_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 74
        rule_74(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

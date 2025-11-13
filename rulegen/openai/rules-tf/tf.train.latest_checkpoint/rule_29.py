import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# latest_filename, if provided, should not be an einsum pattern token (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != 6, And(And(And(And(And(v["arg1_value"] != 0, v["arg1_value"] != 1), v["arg1_value"] != 2), v["arg1_value"] != 3), v["arg1_value"] != 4), v["arg1_value"] != 5), True)) if n else
          If(v["arg1_value"] != 6, And(And(And(And(And(v["arg1_value"] != 0, v["arg1_value"] != 1), v["arg1_value"] != 2), v["arg1_value"] != 3), v["arg1_value"] != 4), v["arg1_value"] != 5), True))
)

def rule_29_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))

        # Constraints for rule 29
        rule_29(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_value': arg1['value']}, neg)

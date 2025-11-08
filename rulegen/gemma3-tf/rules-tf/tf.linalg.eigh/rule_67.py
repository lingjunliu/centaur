import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Name must be a valid string. (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, True, If(v["arg1_value"] == 1, True, If(v["arg1_value"] == 2, True, If(v["arg1_value"] == 3, True, If(v["arg1_value"] == 4, True, If(v["arg1_value"] == 5, True, If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 8, True, If(v["arg1_value"] == 9, True, If(v["arg1_value"] == 10, True, False)))))))))))) if n else
          If(v["arg1_value"] == 0, True, If(v["arg1_value"] == 1, True, If(v["arg1_value"] == 2, True, If(v["arg1_value"] == 3, True, If(v["arg1_value"] == 4, True, If(v["arg1_value"] == 5, True, If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 8, True, If(v["arg1_value"] == 9, True, If(v["arg1_value"] == 10, True, False))))))))))))
)

def rule_67_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 67
        rule_67(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_value': arg1['value']}, neg)

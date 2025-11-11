import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# path may be an activation token string (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (1 + 1), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 12), v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg1_value"] == 19), v["arg1_value"] == 20))) for i in range(6)])) if n else
          Or([And(i < (1 + 1), (Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 12), v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg1_value"] == 19), v["arg1_value"] == 20))) for i in range(6)]))
)

def rule_13_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 13
        rule_13(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_value': arg1['value']}, neg)

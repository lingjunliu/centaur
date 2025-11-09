import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# activation-like names must not clash with reduction names (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If((Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 12), v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg1_value"] == 19), v["arg1_value"] == 20)), (And(And(And(v["arg1_value"] != 7, v["arg1_value"] != 8), v["arg1_value"] != 9), v["arg1_value"] != 10)), True)) if n else
          If((Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 11, v["arg1_value"] == 12), v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg1_value"] == 19), v["arg1_value"] == 20)), (And(And(And(v["arg1_value"] != 7, v["arg1_value"] != 8), v["arg1_value"] != 9), v["arg1_value"] != 10)), True))
)

def rule_22_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 22
        rule_22(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_value': arg1['value']}, neg)

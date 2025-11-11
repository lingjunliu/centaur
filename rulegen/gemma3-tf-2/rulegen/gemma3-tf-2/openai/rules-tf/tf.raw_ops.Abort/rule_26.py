import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# layout-like operation names require a non-neutral message (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == 25, v["arg2_value"] == 24), v["arg1_value"] != 6, True)) if n else
          If(Or(v["arg2_value"] == 25, v["arg2_value"] == 24), v["arg1_value"] != 6, True))
)

def rule_26_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 26
        rule_26(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

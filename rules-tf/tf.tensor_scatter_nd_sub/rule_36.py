import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check policy parameter using an if-then-else construct for string validity  (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 9, True, If(v["arg1_value"] == 8, True, If(v["arg1_value"] == 10, True, False)))))) if n else
          If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 9, True, If(v["arg1_value"] == 8, True, If(v["arg1_value"] == 10, True, False))))))
)

def rule_36_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 36
        rule_36(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_value': arg1['value']}, neg)

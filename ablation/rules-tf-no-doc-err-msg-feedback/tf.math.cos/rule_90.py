import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check that v_1 should be either true or false, and v_2 is a valid string (Rule 90)

rule_90 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == 21, v["arg2_value"] == 22)))) if n else
          And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == 21, v["arg2_value"] == 22))))
)

def rule_90_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 90
        rule_90(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_90(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

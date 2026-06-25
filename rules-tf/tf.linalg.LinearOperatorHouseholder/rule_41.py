import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# All boolean flags are either true or false, and name is a string (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == True, v["arg2_value"] == False))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 6, v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 11), v["arg3_value"] == 12), v["arg3_value"] == 13), v["arg3_value"] == 14), v["arg3_value"] == 15), v["arg3_value"] == 16), v["arg3_value"] == 17), v["arg3_value"] == 18), v["arg3_value"] == 19), v["arg3_value"] == 20), v["arg3_value"] == 21), v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24), v["arg3_value"] == 25)))) if n else
          And(And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == True, v["arg2_value"] == False))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_value"] == 6, v["arg3_value"] == 7), v["arg3_value"] == 8), v["arg3_value"] == 9), v["arg3_value"] == 10), v["arg3_value"] == 11), v["arg3_value"] == 12), v["arg3_value"] == 13), v["arg3_value"] == 14), v["arg3_value"] == 15), v["arg3_value"] == 16), v["arg3_value"] == 17), v["arg3_value"] == 18), v["arg3_value"] == 19), v["arg3_value"] == 20), v["arg3_value"] == 21), v["arg3_value"] == 22), v["arg3_value"] == 23), v["arg3_value"] == 24), v["arg3_value"] == 25))))
)

def rule_41_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 41
        rule_41(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

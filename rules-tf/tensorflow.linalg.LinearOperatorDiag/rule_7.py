import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check if matrix property hints are valid (Rule 7)

rule_7 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == True, v["arg2_value"] == False))), (Or(v["arg3_value"] == True, v["arg3_value"] == False))), (Or(v["arg4_value"] == True, v["arg4_value"] == False)))) if n else
          And(And(And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == True, v["arg2_value"] == False))), (Or(v["arg3_value"] == True, v["arg3_value"] == False))), (Or(v["arg4_value"] == True, v["arg4_value"] == False))))
)

def rule_7_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)

        # Constraints for rule 7
        rule_7(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_7(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

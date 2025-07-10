import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The inplace argument has correct typing (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] >= 0, v["arg1_value"] <= 1), Or(v["arg2_value"] == True, v["arg2_value"] == False), False)) if n else
          If(And(v["arg1_value"] >= 0, v["arg1_value"] <= 1), Or(v["arg2_value"] == True, v["arg2_value"] == False), False))
)

def rule_80_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)

        # Constraints for rule 80
        rule_80(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

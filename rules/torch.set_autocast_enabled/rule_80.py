import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Validate Autocast Enabled based on Dtype and also the opposite (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg1_value"] == False)) if n else
          If(v["arg1_value"] == True, Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg1_value"] == False))
)

def rule_80_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 80
        rule_80(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

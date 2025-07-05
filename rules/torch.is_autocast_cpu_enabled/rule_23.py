import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# torch.is_autocast_cpu_enabled: if any number is equal to zero, then v_1 is a bool (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 0, Or(v["arg1_value"] == True, v["arg1_value"] == False), Or(v["arg1_value"] == True, v["arg1_value"] == False))) if n else
          If(v["arg2_value"] == 0, Or(v["arg1_value"] == True, v["arg1_value"] == False), Or(v["arg1_value"] == True, v["arg1_value"] == False)))
)

def rule_23_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 23
        rule_23(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

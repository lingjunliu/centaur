import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If v_1 is less than 0, v_2 has to be true (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 0, v["arg2_value"] == True, v["arg2_value"] == False)) if n else
          If(v["arg1_value"] < 0, v["arg2_value"] == True, v["arg2_value"] == False))
)

def rule_63_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg2_value == arg2)

        # Constraints for rule 63
        rule_63(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

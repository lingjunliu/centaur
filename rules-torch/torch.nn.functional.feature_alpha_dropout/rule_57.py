import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If training, the probability is between 0.0 and 1.0, otherwise it must be exactly zero. (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"], And(v["arg1_value"] >= 0.0, v["arg1_value"] <= 1.0), v["arg1_value"] == 0.0)) if n else
          If(v["arg2_value"], And(v["arg1_value"] >= 0.0, v["arg1_value"] <= 1.0), v["arg1_value"] == 0.0))
)

def rule_57_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 57
        rule_57(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

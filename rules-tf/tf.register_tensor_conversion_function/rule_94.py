import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# priority is an int ⊎ float and if it is float its value must not be zero and v1 should not be large number and not be small number (Rule 94)

rule_94 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, False, If(v["arg1_value"] > 10000, False, If(v["arg1_value"] < -10000, False, True)))) if n else
          If(v["arg1_value"] == 0, False, If(v["arg1_value"] > 10000, False, If(v["arg1_value"] < -10000, False, True))))
)

def rule_94_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 94
        rule_94(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_94(solver, {'arg1_value': arg1['value']}, neg)

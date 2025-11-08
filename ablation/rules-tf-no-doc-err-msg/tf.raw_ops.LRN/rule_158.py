import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# beta should be a float value between 0.3 and 0.7 and not equal to 0.5 (Rule 158)

rule_158 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] > 0.3, v["arg1_value"] < 0.7), v["arg1_value"] != 0.5)) if n else
          And(And(v["arg1_value"] > 0.3, v["arg1_value"] < 0.7), v["arg1_value"] != 0.5))
)

def rule_158_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 158
        rule_158(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_158(solver, {'arg1_value': arg1['value']}, neg)

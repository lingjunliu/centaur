import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If n is a float, it should be within the acceptable integer range and also greater than 0 (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg1_value"] > -2147483648.0), (v["arg1_value"] < 2147483647.0)), (v["arg1_value"] > 0)), True, False)) if n else
          If(And(And((v["arg1_value"] > -2147483648.0), (v["arg1_value"] < 2147483647.0)), (v["arg1_value"] > 0)), True, False))
)

def rule_40_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 40
        rule_40(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_value': arg1['value']}, neg)

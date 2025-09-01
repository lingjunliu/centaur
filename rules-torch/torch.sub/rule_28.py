import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If other is number, ensure it is not infinity to avoid numerical instability, check against upper and lower bounds (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(And(-10000 < v["arg1_value"], v["arg1_value"] < 10000)) if n else
          And(-10000 < v["arg1_value"], v["arg1_value"] < 10000))
)

def rule_28_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 28
        rule_28(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_value': arg1['value']}, neg)

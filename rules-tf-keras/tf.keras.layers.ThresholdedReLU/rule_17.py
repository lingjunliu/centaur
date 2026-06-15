import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The threshold parameter theta must be non-negative and bounded within a standard practical range of 1000.0 (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(And(0.0 <= v["arg1_value"], v["arg1_value"] < 1000.0)) if n else
          And(0.0 <= v["arg1_value"], v["arg1_value"] < 1000.0))
)

def rule_17_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 17
        rule_17(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_value': arg1['value']}, neg)

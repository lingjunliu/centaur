import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If mean is a negative large number, stddev needs to be positive small to avoid issues with creating very small and very large values (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < -100000, And(v["arg2_value"] < 10000, v["arg2_value"] > 0), False)) if n else
          If(v["arg1_value"] < -100000, And(v["arg2_value"] < 10000, v["arg2_value"] > 0), False))
)

def rule_76_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 76
        rule_76(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

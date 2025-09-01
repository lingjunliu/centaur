import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Complete num_bits check - must be an integer between 2 and 16 (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If(And((And(2 <= v["arg1_value"], v["arg1_value"] <= 16)), ((v["arg1_value"] - (v["arg1_value"] % 1)) == v["arg1_value"])), True, False)) if n else
          If(And((And(2 <= v["arg1_value"], v["arg1_value"] <= 16)), ((v["arg1_value"] - (v["arg1_value"] % 1)) == v["arg1_value"])), True, False))
)

def rule_51_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 51
        rule_51(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_value': arg1['value']}, neg)

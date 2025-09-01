import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check if num_bits is a valid int32 number with 2 <= num_bits <= 16 to catch out of range values and incorrect type (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg1_value"] >= 2), (v["arg1_value"] <= 16)), ((v["arg1_value"] + 0) == v["arg1_value"])), True, False)) if n else
          If(And(And((v["arg1_value"] >= 2), (v["arg1_value"] <= 16)), ((v["arg1_value"] + 0) == v["arg1_value"])), True, False))
)

def rule_42_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 42
        rule_42(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_value': arg1['value']}, neg)

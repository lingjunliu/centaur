import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# margin must be a valid float and not infinite or NaN and greater or equal to zero and less than a reasonable value, and a positive number (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_value"] != 1.0e38, v["arg1_value"] != -1.0e38), v["arg1_value"] == v["arg1_value"]), v["arg1_value"] >= 0.000000000001), v["arg1_value"] < 1.0e5)) if n else
          And(And(And(And(v["arg1_value"] != 1.0e38, v["arg1_value"] != -1.0e38), v["arg1_value"] == v["arg1_value"]), v["arg1_value"] >= 0.000000000001), v["arg1_value"] < 1.0e5))
)

def rule_37_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 37
        rule_37(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_value': arg1['value']}, neg)

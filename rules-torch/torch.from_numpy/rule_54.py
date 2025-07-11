import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Example with int, greater than 10 and less than 100 and divisible by 2 and not equal to 50 (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] > 10, v["arg1_value"] < 100), v["arg1_value"] / 2 * 2 == v["arg1_value"]), v["arg1_value"] != 50)) if n else
          And(And(And(v["arg1_value"] > 10, v["arg1_value"] < 100), v["arg1_value"] / 2 * 2 == v["arg1_value"]), v["arg1_value"] != 50))
)

def rule_54_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 54
        rule_54(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_value': arg1['value']}, neg)

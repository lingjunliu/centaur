import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If a float is given, it should be finite and not zero. (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] < 1000000000, v["arg1_value"] > -1000000000), v["arg1_value"] != 0)) if n else
          And(And(v["arg1_value"] < 1000000000, v["arg1_value"] > -1000000000), v["arg1_value"] != 0))
)

def rule_46_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 46
        rule_46(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_value': arg1['value']}, neg)

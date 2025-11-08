import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# delta must be non-negative and a finite number, and reasonably small (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] >= 0.0, v["arg1_value"] == v["arg1_value"]), v["arg1_value"] < 100000.0)) if n else
          And(And(v["arg1_value"] >= 0.0, v["arg1_value"] == v["arg1_value"]), v["arg1_value"] < 100000.0))
)

def rule_68_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value']}, neg)

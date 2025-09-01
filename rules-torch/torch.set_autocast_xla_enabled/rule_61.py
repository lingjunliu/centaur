import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Enabled has to be the same as the logical AND with itself (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"], v["arg1_value"] == v["arg1_value"])) if n else
          And(v["arg1_value"], v["arg1_value"] == v["arg1_value"]))
)

def rule_61_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 61
        rule_61(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_value': arg1['value']}, neg)

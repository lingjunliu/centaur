import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# bias should be valid for ConvTranspose3d (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(And(1 <= v["arg1_value"], v["arg1_value"] <= 5)) if n else
          And(1 <= v["arg1_value"], v["arg1_value"] <= 5))
)

def rule_18_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 18
        rule_18(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_value': arg1['value']}, neg)

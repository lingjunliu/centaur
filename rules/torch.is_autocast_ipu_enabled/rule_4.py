import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# torch.is_autocast_ipu_enabled takes no arguments. (Rule 4)

rule_4 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And(v["arg1_value"], True)), (And(v["arg1_value"], False))), (v["arg1_value"] == False))) if n else
          Or(Or((And(v["arg1_value"], True)), (And(v["arg1_value"], False))), (v["arg1_value"] == False)))
)

def rule_4_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 4
        rule_4(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_4(solver, {'arg1_value': arg1['value']}, neg)

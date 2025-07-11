import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If cancel_pending_enqueues exists, its type must be boolean, and it has to be equal to itself (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(And(Or(v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] == v["arg1_value"])) if n else
          And(Or(v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] == v["arg1_value"]))
)

def rule_85_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 85
        rule_85(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_value': arg1['value']}, neg)

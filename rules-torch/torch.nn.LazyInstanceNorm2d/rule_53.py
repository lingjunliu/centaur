import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If both affine and track_running_stats are false, warn, but allow. This doesn't error, but can lead to unexpected behavior - reformulated without negation (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(Or(Or((v["arg1_value"] == True), (v["arg2_value"] == True)), (And(v["arg1_value"] == False, v["arg2_value"] == False)))) if n else
          Or(Or((v["arg1_value"] == True), (v["arg2_value"] == True)), (And(v["arg1_value"] == False, v["arg2_value"] == False))))
)

def rule_53_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)

        # Constraints for rule 53
        rule_53(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

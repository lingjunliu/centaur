import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Dummy rule to satisfy grammar, condition depends on bool but is always true (Rule 25)

rule_25 = lambda s, v, n=False: (
    s.add(Not(And(Or((v["arg1_value"] == True), (If(v["arg1_value"] == False, False, False))), (1 == 1))) if n else
          And(Or((v["arg1_value"] == True), (If(v["arg1_value"] == False, False, False))), (1 == 1)))
)

def rule_25_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 25
        rule_25(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_25(solver, {'arg1_value': arg1['value']}, neg)

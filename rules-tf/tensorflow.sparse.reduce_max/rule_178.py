import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# output_is_sparse is a bool, use XOR operator, different boolean operations and returns it equal value (Rule 178)

rule_178 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_value"], True)), (Or(v["arg1_value"] == False, True))), True == True, False == False)) if n else
          If(And((Or(v["arg1_value"], True)), (Or(v["arg1_value"] == False, True))), True == True, False == False))
)

def rule_178_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 178
        rule_178(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_178(solver, {'arg1_value': arg1['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The boolean is not equal to 1 then it is equivalent to 0, and if it is equal to 1 it is true (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != 1, v["arg1_value"] == 0, v["arg1_value"] == True)) if n else
          If(v["arg1_value"] != 1, v["arg1_value"] == 0, v["arg1_value"] == True))
)

def rule_43_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 43
        rule_43(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_value': arg1['value']}, neg)

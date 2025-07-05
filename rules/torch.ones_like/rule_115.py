import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Make sure layout cannot be any of the reduction functions. Only "strided" is supported (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] != 7, v["arg1_value"] != 8), v["arg1_value"] != 9)) if n else
          And(And(v["arg1_value"] != 7, v["arg1_value"] != 8), v["arg1_value"] != 9))
)

def rule_115_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))

        # Constraints for rule 115
        rule_115(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_value': arg1['value']}, neg)

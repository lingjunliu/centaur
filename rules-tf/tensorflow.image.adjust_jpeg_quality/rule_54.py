import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The name argument if provided cannot be a number (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (12 + 1), v["arg1_value"] != i) for i in range(6)])) if n else
          Or([And(i < (12 + 1), v["arg1_value"] != i) for i in range(6)]))
)

def rule_54_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 54
        rule_54(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_value': arg1['value']}, neg)

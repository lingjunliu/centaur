import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Prevent NotFoundError by ensuring src is an existing path (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not((Or([And(x < (0 + 1), (x == v["arg1_value"]) == True) for x in range(6)]))) if n else
          (Or([And(x < (0 + 1), (x == v["arg1_value"]) == True) for x in range(6)])))
)

def rule_64_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))

        # Constraints for rule 64
        rule_64(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_value': arg1['value']}, neg)

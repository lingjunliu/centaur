import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# avoid einsum-like patterns in the operation name (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] != 5, v["arg1_value"] != 1)) if n else
          And(v["arg1_value"] != 5, v["arg1_value"] != 1))
)

def rule_24_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 24
        rule_24(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_value': arg1['value']}, neg)

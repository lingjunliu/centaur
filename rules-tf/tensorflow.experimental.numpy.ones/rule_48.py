import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Dtype must be a valid dtype integer (1-11 or 13 (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(And(And(1 <= v["arg1_value"], Or(Or(v["arg1_value"] <= 3, v["arg1_value"] == 5), 6 <= v["arg1_value"])), Or(v["arg1_value"] <= 11, v["arg1_value"] == 13))) if n else
          And(And(1 <= v["arg1_value"], Or(Or(v["arg1_value"] <= 3, v["arg1_value"] == 5), 6 <= v["arg1_value"])), Or(v["arg1_value"] <= 11, v["arg1_value"] == 13)))
)

def rule_48_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))

        # Constraints for rule 48
        rule_48(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_value': arg1['value']}, neg)

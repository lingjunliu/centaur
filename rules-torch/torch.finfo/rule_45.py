import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dtype is a floating point type from float16 to complex128 (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] > 5, v["arg1_value"] < 9), True, If(And(v["arg1_value"] > 8, v["arg1_value"] < 11), True, False))) if n else
          If(And(v["arg1_value"] > 5, v["arg1_value"] < 9), True, If(And(v["arg1_value"] > 8, v["arg1_value"] < 11), True, False)))
)

def rule_45_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 45
        rule_45(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_value': arg1['value']}, neg)

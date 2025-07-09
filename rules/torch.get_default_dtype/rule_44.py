import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Dtype must be either float16, float32, float64 or numpy dtype and should be less or equal to float64 and greater or equal than float16. (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 12)), (And(v["arg1_value"] >= 6, v["arg1_value"] <= 8)))) if n else
          And((Or(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 12)), (And(v["arg1_value"] >= 6, v["arg1_value"] <= 8))))
)

def rule_44_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 44
        rule_44(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_value': arg1['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dtype should be float16, float32, float64, complex64 or complex128 index (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, False, If(v["arg1_value"] == 1, False, If(v["arg1_value"] == 2, False, If(v["arg1_value"] == 3, False, If(v["arg1_value"] == 4, False, If(v["arg1_value"] == 5, False, If(v["arg1_value"] == 11, False, True)))))))) if n else
          If(v["arg1_value"] == 0, False, If(v["arg1_value"] == 1, False, If(v["arg1_value"] == 2, False, If(v["arg1_value"] == 3, False, If(v["arg1_value"] == 4, False, If(v["arg1_value"] == 5, False, If(v["arg1_value"] == 11, False, True))))))))
)

def rule_39_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 39
        rule_39(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_value': arg1['value']}, neg)

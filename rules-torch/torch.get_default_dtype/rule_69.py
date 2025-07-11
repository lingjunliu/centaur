import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If set default to float32, get_default returns float32 or numpy dtype. (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 7, Or((v["arg1_value"] == 7), (v["arg1_value"] == 12)), False)) if n else
          If(v["arg1_value"] == 7, Or((v["arg1_value"] == 7), (v["arg1_value"] == 12)), False))
)

def rule_69_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 69
        rule_69(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_value': arg1['value']}, neg)

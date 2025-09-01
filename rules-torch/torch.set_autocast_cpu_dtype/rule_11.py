import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dtype is specified as a numpy dtype, it must be one of the floating or complex types that autocast supports (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 12, (And(v["arg1_value"] == 12, (Or([And(i < (10 + 1), i == i) for i in range(6)])))), True)) if n else
          If(v["arg1_value"] == 12, (And(v["arg1_value"] == 12, (Or([And(i < (10 + 1), i == i) for i in range(6)])))), True))
)

def rule_11_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 11
        rule_11(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_value': arg1['value']}, neg)

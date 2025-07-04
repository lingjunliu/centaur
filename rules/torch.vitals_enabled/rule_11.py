import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# vitals_enabled should return true or false, and not a dtype (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, True, If(v["arg1_value"] == False, True, False))) if n else
          If(v["arg1_value"] == True, True, If(v["arg1_value"] == False, True, False)))
)

def rule_11_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool) or (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 11
        rule_11(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_value': arg1['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# dtype and mode are mutually dependent and limited. If mode="none", dtype can be any; if mode!="none", dtype should be one of float16, float32 and bfloat16 (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 6, True, (Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 13)))) if n else
          If(v["arg1_value"] == 6, True, (Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 13))))
)

def rule_10_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not ((isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)) or isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))

        # Constraints for rule 10
        rule_10(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

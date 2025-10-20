import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If tensor's dtype is float16, then value must be a representable float16 (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And(And(v["arg2_value"] >= -65504.0, v["arg2_value"] <= 65504.0), v["arg2_value"] % 0.00006103515625 == 0), True)) if n else
          If(v["arg1_dtype"] == 6, And(And(v["arg2_value"] >= -65504.0, v["arg2_value"] <= 65504.0), v["arg2_value"] % 0.00006103515625 == 0), True))
)

def rule_34_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 34
        rule_34(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)

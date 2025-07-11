import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If input tensor is float16, threshold and value parameters must be representable as float16 (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And(And(And(v["arg2_value"] >= -65504.0, v["arg2_value"] <= 65504.0), v["arg3_value"] >= -65504.0), v["arg3_value"] <= 65504.0), False)) if n else
          If(v["arg1_dtype"] == 6, And(And(And(v["arg2_value"] >= -65504.0, v["arg2_value"] <= 65504.0), v["arg3_value"] >= -65504.0), v["arg3_value"] <= 65504.0), False))
)

def rule_57_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 57
        rule_57(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

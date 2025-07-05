import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If ReLU fusion is enabled, the out_zero_point must be a valid quantized value (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], Or((And(And(v["arg2_value"] == 1, -128 <= v["arg3_value"]), v["arg3_value"] <= 127)), (And(And(v["arg2_value"] == 5, 0 <= v["arg3_value"]), v["arg3_value"] <= 255))), False)) if n else
          If(v["arg1_value"], Or((And(And(v["arg2_value"] == 1, -128 <= v["arg3_value"]), v["arg3_value"] <= 127)), (And(And(v["arg2_value"] == 5, 0 <= v["arg3_value"]), v["arg3_value"] <= 255))), False))
)

def rule_79_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 79
        rule_79(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

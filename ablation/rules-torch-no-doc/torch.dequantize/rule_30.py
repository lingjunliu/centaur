import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If scale and zero point are not tensors, ensure they are within allowed ranges (min/max values of the input quantized type (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(If((v["arg3_value"] == 1), (And(And(And(v["arg1_value"] >= -128, v["arg1_value"] <= 127), v["arg2_value"] >= -128), v["arg2_value"] <= 127)), True)) if n else
          If((v["arg3_value"] == 1), (And(And(And(v["arg1_value"] >= -128, v["arg1_value"] <= 127), v["arg2_value"] >= -128), v["arg2_value"] <= 127)), True))
)

def rule_30_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 30
        rule_30(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

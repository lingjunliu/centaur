import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dim must be an integer type if specified (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, True, If(v["arg1_value"] == 1, True, If(v["arg1_value"] == 2, True, If(v["arg1_value"] == 3, True, If(v["arg1_value"] == 4, True, If(v["arg1_value"] == 5, True, If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 8, True, True)))))))))) if n else
          If(v["arg1_value"] == 0, True, If(v["arg1_value"] == 1, True, If(v["arg1_value"] == 2, True, If(v["arg1_value"] == 3, True, If(v["arg1_value"] == 4, True, If(v["arg1_value"] == 5, True, If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 8, True, True))))))))))
)

def rule_39_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating)) or isinstance(arg1, bool) or isinstance(arg1, str) or (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 39
        rule_39(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_value': arg1['value']}, neg)

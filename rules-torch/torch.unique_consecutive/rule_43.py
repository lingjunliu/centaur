import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dim must be a valid value if specified and not none (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, True, If(v["arg1_value"] == 1, True, If(v["arg1_value"] == 2, True, If(v["arg1_value"] == 3, True, If(v["arg1_value"] == 4, True, If(v["arg1_value"] == 5, True, If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 8, True, If(v["arg1_value"] > 0, If(And(v["arg1_value"] >= (0 - v["arg2_ndim"]), v["arg1_value"] < v["arg2_ndim"]), True, False), If(v["arg1_value"] < 0, If(And(v["arg1_value"] >= (0 - v["arg2_ndim"]), v["arg1_value"] < v["arg2_ndim"]), True, False), True)))))))))))) if n else
          If(v["arg1_value"] == 0, True, If(v["arg1_value"] == 1, True, If(v["arg1_value"] == 2, True, If(v["arg1_value"] == 3, True, If(v["arg1_value"] == 4, True, If(v["arg1_value"] == 5, True, If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 8, True, If(v["arg1_value"] > 0, If(And(v["arg1_value"] >= (0 - v["arg2_ndim"]), v["arg1_value"] < v["arg2_ndim"]), True, False), If(v["arg1_value"] < 0, If(And(v["arg1_value"] >= (0 - v["arg2_ndim"]), v["arg1_value"] < v["arg2_ndim"]), True, False), True))))))))))))
)

def rule_43_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating)) or isinstance(arg1, bool) or isinstance(arg1, str) or (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType))):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 43
        rule_43(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

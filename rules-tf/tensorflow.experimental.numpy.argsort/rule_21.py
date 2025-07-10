import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If 'axis' is specified and input tensor is specified then the 'axis' should be a valid integer (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, v["arg2_value"] == v["arg2_value"], False)) if n else
          If(v["arg1_ndim"] > 0, v["arg2_value"] == v["arg2_value"], False))
)

def rule_21_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating)) or isinstance(arg2, bool) or isinstance(arg2, str) or (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 21
        rule_21(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

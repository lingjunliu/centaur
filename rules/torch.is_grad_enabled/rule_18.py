import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if a tensor and a dtype are given, and the tensor has more than 0 dims, then the dtype cannot be 0,1, or 2 (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, And(And(v["arg2_value"] != 0, v["arg2_value"] != 1), v["arg2_value"] != 2), False)) if n else
          If(v["arg1_ndim"] > 0, And(And(v["arg2_value"] != 0, v["arg2_value"] != 1), v["arg2_value"] != 2), False))
)

def rule_18_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 18
        rule_18(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

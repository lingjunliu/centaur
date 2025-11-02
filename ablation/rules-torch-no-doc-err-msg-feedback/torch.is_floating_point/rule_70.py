import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If v_1 is a tensor and v_2 is a dtype, and if the v_2's value is 7 or 8, then the v_1's number of dimension must be less than 5 (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg1_ndim"] < 5, True)) if n else
          If(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg1_ndim"] < 5, True))
)

def rule_70_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 70
        rule_70(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

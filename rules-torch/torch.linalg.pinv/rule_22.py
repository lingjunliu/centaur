import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If rcond is a tensor, it should be of float or double type, and its ndim should be less than or equal to input tensor's ndim - 2 (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 0, And((Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8)), (v["arg2_ndim"] <= (v["arg1_ndim"] - 2))), True)) if n else
          If(v["arg2_ndim"] > 0, And((Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8)), (v["arg2_ndim"] <= (v["arg1_ndim"] - 2))), True))
)

def rule_22_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 22
        rule_22(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)

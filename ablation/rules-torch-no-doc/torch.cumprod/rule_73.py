import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input dtype must be valid for cumprod and output dtype, if specified, must be same as input. The `dim` argument must be a valid dimension for the input tensor. (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(And(And((And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11)), (And(v["arg2_value"] >= (0 - v["arg1_ndim"]), v["arg2_value"] < v["arg1_ndim"]))), (If(v["arg3_value"] != 12, v["arg1_dtype"] == v["arg3_value"], True)))) if n else
          And(And((And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11)), (And(v["arg2_value"] >= (0 - v["arg1_ndim"]), v["arg2_value"] < v["arg1_ndim"]))), (If(v["arg3_value"] != 12, v["arg1_dtype"] == v["arg3_value"], True))))
)

def rule_73_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 73
        rule_73(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

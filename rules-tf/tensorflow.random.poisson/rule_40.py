import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# shape must be 1D, lam tensor dtype should not be int64 and dtype of the generated tensor should be valid (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == 1, If(v["arg2_dtype"] == 4, False, (Or(Or(Or(v["arg3_value"] == 3, v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8))))) if n else
          And(v["arg1_ndim"] == 1, If(v["arg2_dtype"] == 4, False, (Or(Or(Or(v["arg3_value"] == 3, v["arg3_value"] == 6), v["arg3_value"] == 7), v["arg3_value"] == 8)))))
)

def rule_40_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 40
        rule_40(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dtype is specified and it is float, double, cfloat, or cdouble, then out must be a tensor (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg2_ndim"] > -1, True)) if n else
          If(Or(Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg2_ndim"] > -1, True))
)

def rule_115_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 115
        rule_115(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

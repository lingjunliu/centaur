import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# When reduction is not none, the output tensor is scalar, or 1D with single element, if weight is specified  (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] != 6, Or((v["arg1_ndim"] == 0), (And(v["arg3_ndim"] > 0, (And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 1))))), True)) if n else
          If(v["arg2_value"] != 6, Or((v["arg1_ndim"] == 0), (And(v["arg3_ndim"] > 0, (And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 1))))), True))
)

def rule_74_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 74
        rule_74(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)

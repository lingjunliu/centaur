import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If keep_dims is false, the rank of the output is reduced by the number of reduction axes. (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == False, v["arg4_ndim"] == v["arg2_ndim"] - Select(v["arg3_shape"], 0), True)) if n else
          If(v["arg1_value"] == False, v["arg4_ndim"] == v["arg2_ndim"] - Select(v["arg3_shape"], 0), True))
)

def rule_26_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 26
        rule_26(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg4_ndim': arg4['ndim']}, neg)

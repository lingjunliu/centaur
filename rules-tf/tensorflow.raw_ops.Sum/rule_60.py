import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If keep_dims is false, the rank of the output is less than or equal to the rank of the input. (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == False, v["arg1_ndim"] >= (v["arg1_ndim"] - Select(v["arg3_shape"], 0)), True)) if n else
          If(v["arg2_value"] == False, v["arg1_ndim"] >= (v["arg1_ndim"] - Select(v["arg3_shape"], 0)), True))
)

def rule_60_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Bool('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 60
        rule_60(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)

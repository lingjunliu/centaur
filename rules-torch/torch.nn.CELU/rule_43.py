import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if alpha and a tensor value are close to zero, the gradient should be close to zero (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] < 0.001, v["arg1_value"] > -0.001), Or([And(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) < 0.001, Select(v["arg2_shape"], i) > -0.001)) for i in range(6)]), True)) if n else
          If(And(v["arg1_value"] < 0.001, v["arg1_value"] > -0.001), Or([And(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) < 0.001, Select(v["arg2_shape"], i) > -0.001)) for i in range(6)]), True))
)

def rule_43_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 43
        rule_43(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

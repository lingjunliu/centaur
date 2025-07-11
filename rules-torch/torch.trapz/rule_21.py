import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# y and x must have the same length along dim if x is provided and dim is valid. Otherwise, if x is not given (i.e., shape(x,i (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 0) for i in range(6)]), True, Or((And((And(-1 * v["arg1_ndim"] <= v["arg3_value"], v["arg3_value"] <= v["arg1_ndim"] - 1)), Select(v["arg1_shape"], v["arg3_value"]) == Select(v["arg2_shape"], v["arg3_value"]))), (Or((-1 * v["arg1_ndim"] > v["arg3_value"]), (v["arg3_value"] > v["arg1_ndim"] - 1)))))) if n else
          If(Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 0) for i in range(6)]), True, Or((And((And(-1 * v["arg1_ndim"] <= v["arg3_value"], v["arg3_value"] <= v["arg1_ndim"] - 1)), Select(v["arg1_shape"], v["arg3_value"]) == Select(v["arg2_shape"], v["arg3_value"]))), (Or((-1 * v["arg1_ndim"] > v["arg3_value"]), (v["arg3_value"] > v["arg1_ndim"] - 1))))))
)

def rule_21_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 21
        rule_21(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

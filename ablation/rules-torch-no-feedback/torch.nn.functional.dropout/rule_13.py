import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If training is true and p > 0, at least one element of input tensor should be non-zero (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"], v["arg2_value"] > 0), Or([And(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) > 0, Select(v["arg1_range"], 1) > 0)) for i in range(6)]), True)) if n else
          If(And(v["arg3_value"], v["arg2_value"] > 0), Or([And(i < (v["arg1_ndim"] - 1 + 1), And(Select(v["arg1_shape"], i) > 0, Select(v["arg1_range"], 1) > 0)) for i in range(6)]), True))
)

def rule_13_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 13
        rule_13(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if float v_1 is not equal to 0 and tensor v_2 has positive dimension and max > 0, then the shape of its dimension 0 should be greater or equal to float v_1 (Rule 505)

rule_505 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] != 0, v["arg2_ndim"] > 0), Select(v["arg2_range"], 1) > 0), Select(v["arg2_shape"], 0) >= v["arg1_value"], False)) if n else
          If(And(And(v["arg1_value"] != 0, v["arg2_ndim"] > 0), Select(v["arg2_range"], 1) > 0), Select(v["arg2_shape"], 0) >= v["arg1_value"], False))
)

def rule_505_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 505
        rule_505(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_505(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range']}, neg)

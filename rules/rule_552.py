import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If str v_1 is not "none" and tensor v_2 is not empty, then its maximum has to be larger than the minimum by float v_3 value  (Rule 552)

rule_552 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] != 6, (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), (Select(v["arg2_range"], 1) + (0 - Select(v["arg2_range"], 0))) > v["arg3_value"], False)) if n else
          If(And(v["arg1_value"] != 6, (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), (Select(v["arg2_range"], 1) + (0 - Select(v["arg2_range"], 0))) > v["arg3_value"], False))
)

def rule_552_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)

        # Constraints for rule 552
        rule_552(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_552(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

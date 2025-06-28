import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if string v_1 is one of 'mean','sum','max', and tensor v_2 is non-empty, then tensor v_2 should have all positive numbers (Rule 391)

rule_391 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9)), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]), False)) if n else
          If(And((Or(Or(v["arg1_value"] == 7, v["arg1_value"] == 8), v["arg1_value"] == 9)), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]), False))
)

def rule_391_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 391
        rule_391(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_391(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

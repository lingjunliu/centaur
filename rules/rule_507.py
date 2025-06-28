import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If boolean v_1 equals to true then shape of dimension 0 from tensor v_2 must be not equal to integer v_3 multiplied with the maximum of tensor v_2 (Rule 507)

rule_507 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], Select(v["arg2_shape"], 0) != (v["arg3_value"] * Select(v["arg2_range"], 1)), False)) if n else
          If(v["arg1_value"], Select(v["arg2_shape"], 0) != (v["arg3_value"] * Select(v["arg2_range"], 1)), False))
)

def rule_507_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 507
        rule_507(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_507(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

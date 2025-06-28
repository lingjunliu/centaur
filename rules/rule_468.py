import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If tensor v_1 contains shape in its all dimensions, and string v_2 is 'tanh', then v_3 bool should be false when the result of the sum between maximum and minimum element of the tensor v_1 is positive  (Rule 468)

rule_468 = lambda s, v, n=False: (
    s.add(Not(If(And((And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), v["arg2_value"] == 11), If(Select(v["arg1_range"], 1) + Select(v["arg1_range"], 0) > 0, False, v["arg3_value"]), False)) if n else
          If(And((And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), v["arg2_value"] == 11), If(Select(v["arg1_range"], 1) + Select(v["arg1_range"], 0) > 0, False, v["arg3_value"]), False))
)

def rule_468_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))
        solver.add(arg3_value == arg3)

        # Constraints for rule 468
        rule_468(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_468(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

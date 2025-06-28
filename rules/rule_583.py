import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if integer v_1 is less than 0 and it is a valid index of tensor v_2 and bool v_3 is false, then shape value for the given dimension v_1 for tensor v_2 has to be equal to 1 (Rule 583)

rule_583 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_value"] < 0, v["arg1_value"] >= (0 - v["arg2_ndim"])), v["arg2_ndim"] > 0), v["arg3_value"] == False), Select(v["arg2_shape"], v["arg1_value"]) == 1, False)) if n else
          If(And(And(And(v["arg1_value"] < 0, v["arg1_value"] >= (0 - v["arg2_ndim"])), v["arg2_ndim"] > 0), v["arg3_value"] == False), Select(v["arg2_shape"], v["arg1_value"]) == 1, False))
)

def rule_583_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)

        # Constraints for rule 583
        rule_583(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_583(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

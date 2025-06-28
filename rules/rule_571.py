import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if string v_1 contains 'b' and the shape value of tensor v_2 at index 0 is greater than one then number v_3 must be smaller than 0 (Rule 571)

rule_571 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_value"] == 3, v["arg1_value"] == 5)), v["arg2_ndim"] > 0), And(Select(v["arg2_shape"], 0) < v["arg3_value"], v["arg3_value"] < 0), False)) if n else
          If(And((Or(v["arg1_value"] == 3, v["arg1_value"] == 5)), v["arg2_ndim"] > 0), And(Select(v["arg2_shape"], 0) < v["arg3_value"], v["arg3_value"] < 0), False))
)

def rule_571_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg3_value = Real('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)

        # Constraints for rule 571
        rule_571(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_571(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

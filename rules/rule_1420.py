import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if the length of a tuple of integers v_1 is greater than 0, and string v_2 is equal to "tanh" then elements of tuple v_1 have to smaller than shape of tensor v_3 for those dimensions. (Rule 1420)

rule_1420 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] > 0, v["arg2_value"] == 11), And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) < Select(v["arg3_shape"], i)) for i in range(6)]), False)) if n else
          If(And(v["arg1_length"] > 0, v["arg2_value"] == 11), And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) < Select(v["arg3_shape"], i)) for i in range(6)]), False))
)

def rule_1420_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_string_values.index(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 1420
        rule_1420(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1420(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)

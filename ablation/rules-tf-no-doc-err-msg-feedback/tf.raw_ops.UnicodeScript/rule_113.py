import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the v_1 is true then the shape of v_2 tensor dimension 0 and the length of list v_3 must be equal (Rule 113)

rule_113 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], Select(v["arg2_shape"], 0) == v["arg3_length"], True)) if n else
          If(v["arg1_value"], Select(v["arg2_shape"], 0) == v["arg3_length"], True))
)

def rule_113_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 113
        rule_113(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_113(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_length': arg3['length']}, neg)

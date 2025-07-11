import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

#  check if the size of tensor is the same with integer and compare with list of integers and int is larger than 5 and there is at least on negative value in the list (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(And(And(And(Select(v["arg1_shape"], 0) == v["arg2_value"], v["arg3_length"] == 4), v["arg2_value"] > 5), (Or([And(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) < 0) for i in range(6)])))) if n else
          And(And(And(Select(v["arg1_shape"], 0) == v["arg2_value"], v["arg3_length"] == 4), v["arg2_value"] > 5), (Or([And(i < (v["arg3_length"] - 1 + 1), Select(v["arg3_values"], i) < 0) for i in range(6)]))))
)

def rule_93_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 93
        rule_93(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)

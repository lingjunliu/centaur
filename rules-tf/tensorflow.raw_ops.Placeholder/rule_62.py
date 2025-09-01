import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If shape has a non-zero length and dtype is either bool or int8 shape values should be positive.If dtype is string, then name can not be named as 'sum' (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] > 0, (Or(v["arg2_value"] == 0, v["arg2_value"] == 1))), And((And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) > 0) for i in range(6)])), If(v["arg2_value"] == 12, v["arg3_value"] != 7, True)), True)) if n else
          If(And(v["arg1_length"] > 0, (Or(v["arg2_value"] == 0, v["arg2_value"] == 1))), And((And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) > 0) for i in range(6)])), If(v["arg2_value"] == 12, v["arg3_value"] != 7, True)), True))
)

def rule_62_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 62
        rule_62(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

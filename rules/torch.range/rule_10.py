import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If dtype is int64, start, end, and step should also be integers. (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] == 4, And(And((And(v["arg1_value"] == v["arg1_value"], Or([And(i < (10000 + 1), v["arg1_value"] == i) for i in range(6)]))), (And(v["arg2_value"] == v["arg2_value"], Or([And(i < (10000 + 1), v["arg2_value"] == i) for i in range(6)])))), (And(v["arg3_value"] == v["arg3_value"], Or([And(i < (10000 + 1), v["arg3_value"] == i) for i in range(6)])))), False)) if n else
          If(v["arg4_value"] == 4, And(And((And(v["arg1_value"] == v["arg1_value"], Or([And(i < (10000 + 1), v["arg1_value"] == i) for i in range(6)]))), (And(v["arg2_value"] == v["arg2_value"], Or([And(i < (10000 + 1), v["arg2_value"] == i) for i in range(6)])))), (And(v["arg3_value"] == v["arg3_value"], Or([And(i < (10000 + 1), v["arg3_value"] == i) for i in range(6)])))), False))
)

def rule_10_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Real('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))

        # Constraints for rule 10
        rule_10(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the dtype is complex64 or complex128 and validate indices is true then batch_shape cannot be larger than certain value to avoid running out of memory. (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_value"] == 9, v["arg1_value"] == 10), If(v["arg3_value"] == True, If(v["arg2_length"] > 0, (And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) < 1000) for i in range(6)])), True), True), True)) if n else
          If(Or(v["arg1_value"] == 9, v["arg1_value"] == 10), If(v["arg3_value"] == True, If(v["arg2_length"] > 0, (And([Implies(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) < 1000) for i in range(6)])), True), True), True))
)

def rule_76_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == arg3)

        # Constraints for rule 76
        rule_76(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value']}, neg)

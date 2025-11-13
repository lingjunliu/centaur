import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

#  if there are elements and they are not positive then the dtype must be 11 (Rule 110)

rule_110 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_length"] > 0), (Or([And(v_3 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_3) <= 0) for v_3 in range(6)]))), v["arg2_value"] == 11, True)) if n else
          If(And((v["arg1_length"] > 0), (Or([And(v_3 < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], v_3) <= 0) for v_3 in range(6)]))), v["arg2_value"] == 11, True))
)

def rule_110_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 110
        rule_110(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_110(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value']}, neg)

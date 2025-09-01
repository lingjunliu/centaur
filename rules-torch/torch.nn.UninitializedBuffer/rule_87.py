import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# dtype and size should match up. If there are big dimensions, the type needs to be able to hold it (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) > 2147483647) for i in range(6)]), Or(Or(v["arg2_value"] == 8, v["arg2_value"] == 4), v["arg2_value"] == 10), True)) if n else
          If(Or([And(i < (v["arg1_length"] - 1 + 1), Select(v["arg1_values"], i) > 2147483647) for i in range(6)]), Or(Or(v["arg2_value"] == 8, v["arg2_value"] == 4), v["arg2_value"] == 10), True))
)

def rule_87_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
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

        # Constraints for rule 87
        rule_87(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value']}, neg)

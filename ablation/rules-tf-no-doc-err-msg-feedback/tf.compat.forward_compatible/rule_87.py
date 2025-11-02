import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dtype of v_1 is int, v_2 should be > 10, else if dtype of v_1 is float, v_2 should be < 10 (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg2_value"] > 10, If(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg2_value"] < 10, True))) if n else
          If(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg2_value"] > 10, If(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg2_value"] < 10, True)))
)

def rule_87_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 87
        rule_87(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)

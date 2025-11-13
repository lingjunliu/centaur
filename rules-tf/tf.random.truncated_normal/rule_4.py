import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# mean and stddev must have the same dtype as dtype, if dtype is specified. (Rule 4)

rule_4 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 7, (And(v["arg1_value"] == 7, v["arg2_value"] == 7)), If(v["arg3_value"] == 8, (And(v["arg1_value"] == 8, v["arg2_value"] == 8)), If(v["arg3_value"] == 9, (And(v["arg1_value"] == 9, v["arg2_value"] == 9)), If(v["arg3_value"] == 10, (And(v["arg1_value"] == 10, v["arg2_value"] == 10)), True))))) if n else
          If(v["arg3_value"] == 7, (And(v["arg1_value"] == 7, v["arg2_value"] == 7)), If(v["arg3_value"] == 8, (And(v["arg1_value"] == 8, v["arg2_value"] == 8)), If(v["arg3_value"] == 9, (And(v["arg1_value"] == 9, v["arg2_value"] == 9)), If(v["arg3_value"] == 10, (And(v["arg1_value"] == 10, v["arg2_value"] == 10)), True)))))
)

def rule_4_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 4
        rule_4(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_4(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

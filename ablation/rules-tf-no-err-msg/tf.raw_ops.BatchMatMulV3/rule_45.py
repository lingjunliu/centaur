import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Tout must be bfloat16/half/float32/float64/complex64/complex128 if grad_x or grad_y are enabled (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"], v["arg3_value"]), Or(Or(Or(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), True)) if n else
          If(Or(v["arg2_value"], v["arg3_value"]), Or(Or(Or(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 1), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), True))
)

def rule_45_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 45
        rule_45(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is int and v_2 is not tensor then the promoted dtype v_2 must be in the int or bool range. (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(If(And(1 <= v["arg1_value"], v["arg1_value"] <= 5), Or(v["arg2_dtype"] == 0, (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))), False)) if n else
          If(And(1 <= v["arg1_value"], v["arg1_value"] <= 5), Or(v["arg2_dtype"] == 0, (And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5))), False))
)

def rule_48_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 48
        rule_48(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)

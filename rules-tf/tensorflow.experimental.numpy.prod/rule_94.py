import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the intended resulting type is int, the input tensor type should be integer as well (Rule 94)

rule_94 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] >= 1, v["arg1_value"] <= 5), And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 5), True)) if n else
          If(And(v["arg1_value"] >= 1, v["arg1_value"] <= 5), And(v["arg2_dtype"] >= 1, v["arg2_dtype"] <= 5), True))
)

def rule_94_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 94
        rule_94(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_94(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)

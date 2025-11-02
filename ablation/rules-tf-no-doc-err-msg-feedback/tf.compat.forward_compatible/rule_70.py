import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Ensure that the length of tuple v_1 is equal to the number of dimensions of v_2 and the dtype of v_3 is an integer. (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] == v["arg2_ndim"], (Or(Or(Or(Or(v["arg3_value"] == 1, v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5)))) if n else
          And(v["arg1_length"] == v["arg2_ndim"], (Or(Or(Or(Or(v["arg3_value"] == 1, v["arg3_value"] == 2), v["arg3_value"] == 3), v["arg3_value"] == 4), v["arg3_value"] == 5))))
)

def rule_70_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 70
        rule_70(solver, {'arg1_length': arg1_length, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_length': arg1['length'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)

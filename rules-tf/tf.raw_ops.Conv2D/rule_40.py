import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When padding is EXPLICIT and data_format is NCHW, explicit_paddings length must be 2 * (rank(input (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 30, v["arg4_value"] == 34), v["arg2_length"] == 2 * (v["arg3_ndim"] - 2), True)) if n else
          If(And(v["arg1_value"] == 30, v["arg4_value"] == 34), v["arg2_length"] == 2 * (v["arg3_ndim"] - 2), True))
)

def rule_40_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')
        arg3_ndim = Int('arg3_ndim')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 40
        rule_40(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)

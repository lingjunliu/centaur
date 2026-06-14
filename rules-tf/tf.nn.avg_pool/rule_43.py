import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if data_format starts with NC, then N+2 ksize elements must be specified, else N must be specified (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(If((Or(Or(v["arg2_value"] == 32, v["arg2_value"] == 34), v["arg2_value"] == 30)), v["arg3_length"] == v["arg1_ndim"], v["arg3_length"] == v["arg1_ndim"] - 2)) if n else
          If((Or(Or(v["arg2_value"] == 32, v["arg2_value"] == 34), v["arg2_value"] == 30)), v["arg3_length"] == v["arg1_ndim"], v["arg3_length"] == v["arg1_ndim"] - 2))
)

def rule_43_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 43
        rule_43(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)

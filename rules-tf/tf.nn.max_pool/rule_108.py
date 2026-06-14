import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# data_format is valid when tensor rank is 4 (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 4, (Or(v["arg1_value"] == 31, v["arg1_value"] == 32)), (v["arg2_ndim"] != 4))) if n else
          If(v["arg2_ndim"] == 4, (Or(v["arg1_value"] == 31, v["arg1_value"] == 32)), (v["arg2_ndim"] != 4)))
)

def rule_108_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 108
        rule_108(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

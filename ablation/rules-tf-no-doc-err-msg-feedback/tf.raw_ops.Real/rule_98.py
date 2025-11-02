import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If padding is causal, input dimension must be 3, and output will also have same dimension (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 23, And(v["arg2_ndim"] == 3, v["arg3_ndim"] == 3), True)) if n else
          If(v["arg1_value"] == 23, And(v["arg2_ndim"] == 3, v["arg3_ndim"] == 3), True))
)

def rule_98_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 98
        rule_98(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim']}, neg)

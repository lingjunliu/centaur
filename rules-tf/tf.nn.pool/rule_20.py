import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# data_format check for N=3 (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 5, Or(Or(Or(v["arg1_value"] == 35, v["arg1_value"] == 36), v["arg1_value"] == 24), v["arg1_value"] == 25), True)) if n else
          If(v["arg2_ndim"] == 5, Or(Or(Or(v["arg1_value"] == 35, v["arg1_value"] == 36), v["arg1_value"] == 24), v["arg1_value"] == 25), True))
)

def rule_20_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 20
        rule_20(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

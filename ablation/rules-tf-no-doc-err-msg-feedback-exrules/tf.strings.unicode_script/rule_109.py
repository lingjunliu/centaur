import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input string v_1 is one of "elu", "selu", "gelu", or "swish", then ndim of tensor v_2 must be greater than or equal to 2 (Rule 109)

rule_109 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(v["arg1_value"] == 15, v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg2_ndim"] >= 2, True)) if n else
          If(Or(Or(Or(v["arg1_value"] == 15, v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg2_ndim"] >= 2, True))
)

def rule_109_func(arg1, arg2, solver=None, neg=False):
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
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 109
        rule_109(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_109(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

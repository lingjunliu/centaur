import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is a string and v_2 is a tensor, then v_1 must be channels_first or channels_last, or none, or valid or same or causal (Rule 103)

rule_103 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(Or(Or(v["arg1_value"] == 25, v["arg1_value"] == 24), v["arg1_value"] == 6), v["arg1_value"] == 21), v["arg1_value"] == 22), v["arg1_value"] == 23)), v["arg2_ndim"] > 0)) if n else
          And((Or(Or(Or(Or(Or(v["arg1_value"] == 25, v["arg1_value"] == 24), v["arg1_value"] == 6), v["arg1_value"] == 21), v["arg1_value"] == 22), v["arg1_value"] == 23)), v["arg2_ndim"] > 0))
)

def rule_103_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 103
        rule_103(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

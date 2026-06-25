import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the tensor has 4 or 5 dimensions, and name is channels_first or channels_last, the second dimension must be 1 or 3 (Rule 23)

rule_23 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_ndim"] == 4, v["arg1_ndim"] == 5)), (Or(v["arg2_value"] == 25, v["arg2_value"] == 24))), Or(Select(v["arg1_shape"], 1) == 1, Select(v["arg1_shape"], 1) == 3), True)) if n else
          If(And((Or(v["arg1_ndim"] == 4, v["arg1_ndim"] == 5)), (Or(v["arg2_value"] == 25, v["arg2_value"] == 24))), Or(Select(v["arg1_shape"], 1) == 1, Select(v["arg1_shape"], 1) == 3), True))
)

def rule_23_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 23
        rule_23(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_23(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

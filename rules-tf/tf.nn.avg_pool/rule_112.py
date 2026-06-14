import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If rank is 3 and data_format is NWC or NCW, then dimension 0 must be greater than 0, dimension 1 must be greater than 0, and dimension 2 must be greater than 0. (Rule 112)

rule_112 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] == 3, (Or(v["arg2_value"] == 29, v["arg2_value"] == 30))), And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0), True)) if n else
          If(And(v["arg1_ndim"] == 3, (Or(v["arg2_value"] == 29, v["arg2_value"] == 30))), And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0), True))
)

def rule_112_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 112
        rule_112(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_112(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

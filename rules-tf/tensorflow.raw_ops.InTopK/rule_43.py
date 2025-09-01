import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if targets has shape [batch_size], k should less than equal to classes in predictions and greater than 0 if predictions has more than one dimension  (Rule 43)

rule_43 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_ndim"] == 1, v["arg1_ndim"] > 1), And(v["arg3_value"] <= Select(v["arg1_shape"], 1), v["arg3_value"] > 0), True)) if n else
          If(And(v["arg2_ndim"] == 1, v["arg1_ndim"] > 1), And(v["arg3_value"] <= Select(v["arg1_shape"], 1), v["arg3_value"] > 0), True))
)

def rule_43_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 43
        rule_43(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_43(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Combined parameter validation emphasizing the relationship between filter size and image size (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(v["arg1_value"] == 33, (Or(v["arg1_value"] == 28, v["arg1_value"] == 29))), v["arg2_ndim"] == 3), v["arg3_ndim"] == 4), Select(v["arg3_shape"], 0) > 0), Select(v["arg3_shape"], 1) > 0), Select(v["arg3_shape"], 2) > 0), Select(v["arg3_shape"], 3) > 0), v["arg4_length"] == 4), Select(v["arg2_shape"], 0) <= Select(v["arg3_shape"], 1) + 10), Select(v["arg2_shape"], 1) <= Select(v["arg3_shape"], 2) + 10)) if n else
          And(And(And(And(And(And(And(And(And(And(v["arg1_value"] == 33, (Or(v["arg1_value"] == 28, v["arg1_value"] == 29))), v["arg2_ndim"] == 3), v["arg3_ndim"] == 4), Select(v["arg3_shape"], 0) > 0), Select(v["arg3_shape"], 1) > 0), Select(v["arg3_shape"], 2) > 0), Select(v["arg3_shape"], 3) > 0), v["arg4_length"] == 4), Select(v["arg2_shape"], 0) <= Select(v["arg3_shape"], 1) + 10), Select(v["arg2_shape"], 1) <= Select(v["arg3_shape"], 2) + 10))
)

def rule_102_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_length = Int('arg4_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_length == len(arg4))

        # Constraints for rule 102
        rule_102(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_length': arg4['length']}, neg)

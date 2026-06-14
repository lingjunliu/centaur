import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# spatial dimensions compatibility of input and filters under valid padding (Rule 131)

rule_131 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_ndim"] == 5, v["arg2_ndim"] == 5), v["arg5_length"] == 5), (If(v["arg4_value"] == 21, (If(v["arg3_value"] == 25, And(And(Select(v["arg1_shape"], 2) >= Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 3) >= Select(v["arg2_shape"], 1)), Select(v["arg1_shape"], 4) >= Select(v["arg2_shape"], 2)), And(And(Select(v["arg1_shape"], 1) >= Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 2) >= Select(v["arg2_shape"], 1)), Select(v["arg1_shape"], 3) >= Select(v["arg2_shape"], 2)))), True)))) if n else
          And(And(And(v["arg1_ndim"] == 5, v["arg2_ndim"] == 5), v["arg5_length"] == 5), (If(v["arg4_value"] == 21, (If(v["arg3_value"] == 25, And(And(Select(v["arg1_shape"], 2) >= Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 3) >= Select(v["arg2_shape"], 1)), Select(v["arg1_shape"], 4) >= Select(v["arg2_shape"], 2)), And(And(Select(v["arg1_shape"], 1) >= Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 2) >= Select(v["arg2_shape"], 1)), Select(v["arg1_shape"], 3) >= Select(v["arg2_shape"], 2)))), True))))
)

def rule_131_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, str):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_length = Int('arg5_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))
        solver.add(arg5_length == len(arg5))

        # Constraints for rule 131
        rule_131(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_length': arg5_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_131(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_length': arg5['length']}, neg)

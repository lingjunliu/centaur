import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If padding is a tuple, then the input tensor's last three dimensions must be greater than or equal to the padding (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] == 4, And(Select(v["arg2_shape"], 1) >= Select(v["arg1_values"], 2) + Select(v["arg1_values"], 3), Select(v["arg2_shape"], 2) >= Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1)), If(v["arg2_ndim"] == 5, And(And(Select(v["arg2_shape"], 2) >= Select(v["arg1_values"], 4) + Select(v["arg1_values"], 5), Select(v["arg2_shape"], 3) >= Select(v["arg1_values"], 2) + Select(v["arg1_values"], 3)), Select(v["arg2_shape"], 4) >= Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1)), True))) if n else
          If(v["arg2_ndim"] == 4, And(Select(v["arg2_shape"], 1) >= Select(v["arg1_values"], 2) + Select(v["arg1_values"], 3), Select(v["arg2_shape"], 2) >= Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1)), If(v["arg2_ndim"] == 5, And(And(Select(v["arg2_shape"], 2) >= Select(v["arg1_values"], 4) + Select(v["arg1_values"], 5), Select(v["arg2_shape"], 3) >= Select(v["arg1_values"], 2) + Select(v["arg1_values"], 3)), Select(v["arg2_shape"], 4) >= Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1)), True)))
)

def rule_16_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 16
        rule_16(solver, {'arg1_values': arg1_values, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_values': arg1['values'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

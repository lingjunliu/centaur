import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If bool v_1 is true, and tensor v_2's dimension is positive, then its shape at dimension 0 is different from its shape at last dimension (Rule 489)

rule_489 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], If(v["arg2_ndim"] > 0, Select(v["arg2_shape"], 0) != Select(v["arg2_shape"], v["arg2_ndim"] - 1), False), False)) if n else
          If(v["arg1_value"], If(v["arg2_ndim"] > 0, Select(v["arg2_shape"], 0) != Select(v["arg2_shape"], v["arg2_ndim"] - 1), False), False))
)

def rule_489_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 489
        rule_489(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_489(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

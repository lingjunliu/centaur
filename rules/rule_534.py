import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if the shape of tensor v_1 at first dimension is greater than the shape of the last dimension and string v_2 is constant, then minimum of the tensor v_1 should equal to the maximum of it (Rule 534)

rule_534 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) > Select(v["arg1_shape"], v["arg1_ndim"] - 1)), v["arg2_value"] == 10), Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), False)) if n else
          If(And(And(v["arg1_ndim"] > 0, Select(v["arg1_shape"], 0) > Select(v["arg1_shape"], v["arg1_ndim"] - 1)), v["arg2_value"] == 10), Select(v["arg1_range"], 0) == Select(v["arg1_range"], 1), False))
)

def rule_534_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 534
        rule_534(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_534(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)

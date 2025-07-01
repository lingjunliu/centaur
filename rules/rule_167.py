import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if ndim == 2, then the difference between shape[0] and shape[1] should be less than 10 if string is sum or mean (Rule 167)

rule_167 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] == 2, (Or(v["arg2_value"] == 8, v["arg2_value"] == 7))), And((Select(v["arg1_shape"], 0) - Select(v["arg1_shape"], 1)) < 10, (Select(v["arg1_shape"], 1) - Select(v["arg1_shape"], 0)) < 10), False)) if n else
          If(And(v["arg1_ndim"] == 2, (Or(v["arg2_value"] == 8, v["arg2_value"] == 7))), And((Select(v["arg1_shape"], 0) - Select(v["arg1_shape"], 1)) < 10, (Select(v["arg1_shape"], 1) - Select(v["arg1_shape"], 0)) < 10), False))
)

def rule_167_func(arg1, arg2, solver=None, neg=False):
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
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 167
        rule_167(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_167(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

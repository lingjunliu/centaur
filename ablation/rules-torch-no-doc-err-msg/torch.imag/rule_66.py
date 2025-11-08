import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the string is bilinear, bicubic or trilinear, then the input tensor must have 4, 4, and 5 dimensions respectively, with positive dimensions (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 26, And(v["arg1_ndim"] == 4, And([Implies(i < (3 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), If(v["arg2_value"] == 27, And(v["arg1_ndim"] == 4, And([Implies(i < (3 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), If(v["arg2_value"] == 28, And(v["arg1_ndim"] == 5, And([Implies(i < (4 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), True)))) if n else
          If(v["arg2_value"] == 26, And(v["arg1_ndim"] == 4, And([Implies(i < (3 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), If(v["arg2_value"] == 27, And(v["arg1_ndim"] == 4, And([Implies(i < (3 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), If(v["arg2_value"] == 28, And(v["arg1_ndim"] == 5, And([Implies(i < (4 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), True))))
)

def rule_66_func(arg1, arg2, solver=None, neg=False):
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
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))

        # Constraints for rule 66
        rule_66(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dim is a tuple, and keepdim is set to True and the input has two dimensions, then the first two dimensions should be different and greater than zero. (Rule 144)

rule_144 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"], v["arg1_ndim"] == 2), And(And(Select(v["arg1_shape"], 0) != Select(v["arg1_shape"], 1), Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), True)) if n else
          If(And(v["arg2_value"], v["arg1_ndim"] == 2), And(And(Select(v["arg1_shape"], 0) != Select(v["arg1_shape"], 1), Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 1) > 0), True))
)

def rule_144_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)

        # Constraints for rule 144
        rule_144(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_144(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)

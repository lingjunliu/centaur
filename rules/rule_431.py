import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if integer v_1 is non-negative and smaller than 5, then each shape value of tensor v_2 must be less than v_1 + 10 (Rule 431)

rule_431 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] >= 0, v["arg1_value"] < 5), And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) < v["arg1_value"] + 10) for i in range(6)]), False)) if n else
          If(And(v["arg1_value"] >= 0, v["arg1_value"] < 5), And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) < v["arg1_value"] + 10) for i in range(6)]), False))
)

def rule_431_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 431
        rule_431(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_431(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)

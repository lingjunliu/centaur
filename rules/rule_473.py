import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If float v_1 is less than 0, then for tensor v_2 with dimension >0, all shape have to be the same (Rule 473)

rule_473 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] < 0, v["arg2_ndim"] > 0), And([Implies(i < (v["arg2_ndim"] - 1 + 1), And([Implies(j < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == Select(v["arg2_shape"], j)) for j in range(6)])) for i in range(6)]), False)) if n else
          If(And(v["arg1_value"] < 0, v["arg2_ndim"] > 0), And([Implies(i < (v["arg2_ndim"] - 1 + 1), And([Implies(j < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == Select(v["arg2_shape"], j)) for j in range(6)])) for i in range(6)]), False))
)

def rule_473_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 473
        rule_473(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_473(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)

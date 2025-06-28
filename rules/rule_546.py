import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if boolean v_1 equals true, then shape of every dimension of tensor v_2 must be greater than float v_3 and smaller than float v_4 (Rule 546)

rule_546 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"], v["arg2_ndim"] > 0), And([Implies(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) > v["arg3_value"], Select(v["arg2_shape"], i) < v["arg4_value"])) for i in range(6)]), False)) if n else
          If(And(v["arg1_value"], v["arg2_ndim"] > 0), And([Implies(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) > v["arg3_value"], Select(v["arg2_shape"], i) < v["arg4_value"])) for i in range(6)]), False))
)

def rule_546_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, (float, np.floating)):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Real('arg3_value')
        arg4_value = Real('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)

        # Constraints for rule 546
        rule_546(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_546(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

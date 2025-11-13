import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Dilations values must be positive and less or equal than filter height and width and strides must be positive and less than or equal to input spatial dimensions (Rule 151)

rule_151 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(Select(v["arg1_values"], 1) <= Select(v["arg2_shape"], 0), Select(v["arg1_values"], 2) <= Select(v["arg2_shape"], 1)), Select(v["arg4_values"], 1) <= Select(v["arg3_shape"], 1)), Select(v["arg4_values"], 2) <= Select(v["arg3_shape"], 2)), Select(v["arg4_values"], 1) > 0), Select(v["arg4_values"], 2) > 0)) if n else
          And(And(And(And(And(Select(v["arg1_values"], 1) <= Select(v["arg2_shape"], 0), Select(v["arg1_values"], 2) <= Select(v["arg2_shape"], 1)), Select(v["arg4_values"], 1) <= Select(v["arg3_shape"], 1)), Select(v["arg4_values"], 2) <= Select(v["arg3_shape"], 2)), Select(v["arg4_values"], 1) > 0), Select(v["arg4_values"], 2) > 0))
)

def rule_151_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_values = Array('arg4_values', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])

        # Constraints for rule 151
        rule_151(solver, {'arg1_values': arg1_values, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_values': arg4_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_151(solver, {'arg1_values': arg1['values'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_values': arg4['values']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If integer v_1 is greater than 0, and there exist some dimensions with shape in tensor v_2 smaller than or equal to 5, the string v_3 should equals 'tanh' (Rule 540)

rule_540 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] > 0, (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) <= 5) for i in range(6)]))), v["arg3_value"] == 11, False)) if n else
          If(And(v["arg1_value"] > 0, (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) <= 5) for i in range(6)]))), v["arg3_value"] == 11, False))
)

def rule_540_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == list_of_string_values.index(arg3))

        # Constraints for rule 540
        rule_540(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_540(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)

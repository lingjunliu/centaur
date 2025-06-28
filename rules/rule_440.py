import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If string v_1 equals to 'tanh' and integer v_2 is 1, then the shape for the tensor v_3 must be power of 2 in all of its dimensions. (Rule 440)

rule_440 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 11, v["arg2_value"] == 1), And([Implies(i < (v["arg3_ndim"] - 1 + 1), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg3_shape"], i) == 1, Select(v["arg3_shape"], i) == 2), Select(v["arg3_shape"], i) == 4), Select(v["arg3_shape"], i) == 8), Select(v["arg3_shape"], i) == 16), Select(v["arg3_shape"], i) == 32), Select(v["arg3_shape"], i) == 64), Select(v["arg3_shape"], i) == 128), Select(v["arg3_shape"], i) == 256), Select(v["arg3_shape"], i) == 512), Select(v["arg3_shape"], i) == 1024), Select(v["arg3_shape"], i) == 2048))) for i in range(6)]), False)) if n else
          If(And(v["arg1_value"] == 11, v["arg2_value"] == 1), And([Implies(i < (v["arg3_ndim"] - 1 + 1), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Select(v["arg3_shape"], i) == 1, Select(v["arg3_shape"], i) == 2), Select(v["arg3_shape"], i) == 4), Select(v["arg3_shape"], i) == 8), Select(v["arg3_shape"], i) == 16), Select(v["arg3_shape"], i) == 32), Select(v["arg3_shape"], i) == 64), Select(v["arg3_shape"], i) == 128), Select(v["arg3_shape"], i) == 256), Select(v["arg3_shape"], i) == 512), Select(v["arg3_shape"], i) == 1024), Select(v["arg3_shape"], i) == 2048))) for i in range(6)]), False))
)

def rule_440_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 440
        rule_440(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_440(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)

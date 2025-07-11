import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If local_step can be represented as x*y where x and y both are less than 10, then gradient tensor dimensions must have size x and y, respectively (Rule 92)

rule_92 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(x < (9 + 1), Or([And(y < (9 + 1), x * y == v["arg1_value"]) for y in range(6)])) for x in range(6)]), (If(v["arg2_ndim"] == 2, And(Select(v["arg2_shape"], 0) == x, Select(v["arg2_shape"], 1) == y), False)), False)) if n else
          If(Or([And(x < (9 + 1), Or([And(y < (9 + 1), x * y == v["arg1_value"]) for y in range(6)])) for x in range(6)]), (If(v["arg2_ndim"] == 2, And(Select(v["arg2_shape"], 0) == x, Select(v["arg2_shape"], 1) == y), False)), False))
)

def rule_92_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 92
        rule_92(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_92(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)

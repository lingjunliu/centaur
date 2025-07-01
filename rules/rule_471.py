import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If v_1 integer is between 0 and 3, then there exists at least a dimension in v_2 tensor that is equal or greater than the value given by integer v_1 (Rule 471)

rule_471 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"] >= 0, v["arg1_value"] <= 3), v["arg2_ndim"] > 0), Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) >= v["arg1_value"]) for i in range(6)]), False)) if n else
          If(And(And(v["arg1_value"] >= 0, v["arg1_value"] <= 3), v["arg2_ndim"] > 0), Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) >= v["arg1_value"]) for i in range(6)]), False))
)

def rule_471_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 471
        rule_471(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_471(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)

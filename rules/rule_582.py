import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If tensor v_1 has dimensions equal to v_2 and string v_3 equal to tanh, the shape of the tensor at index v_2 substracted by one should be smaller than max element to that tensor (Rule 582)

rule_582 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_ndim"] == v["arg2_value"], v["arg3_value"] == 11), v["arg1_ndim"] > 0), v["arg2_value"] > 0), Select(v["arg1_shape"], v["arg2_value"] - 1) < Select(v["arg1_range"], 1), False)) if n else
          If(And(And(And(v["arg1_ndim"] == v["arg2_value"], v["arg3_value"] == 11), v["arg1_ndim"] > 0), v["arg2_value"] > 0), Select(v["arg1_shape"], v["arg2_value"] - 1) < Select(v["arg1_range"], 1), False))
)

def rule_582_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_string_values.index(arg3))

        # Constraints for rule 582
        rule_582(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_582(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)

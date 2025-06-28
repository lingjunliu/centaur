import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If boolean v_1 is true and string v_2 equals tanh, then shape 0 of tensor v_3 has to greater than 5 AND smaller than 10 (Rule 568)

rule_568 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_value"], v["arg2_value"] == 11), v["arg3_ndim"] > 0), And(Select(v["arg3_shape"], 0) > 5, Select(v["arg3_shape"], 0) < 10), False)) if n else
          If(And(And(v["arg1_value"], v["arg2_value"] == 11), v["arg3_ndim"] > 0), And(Select(v["arg3_shape"], 0) > 5, Select(v["arg3_shape"], 0) < 10), False))
)

def rule_568_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = String('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_string_values.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 568
        rule_568(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_568(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape']}, neg)

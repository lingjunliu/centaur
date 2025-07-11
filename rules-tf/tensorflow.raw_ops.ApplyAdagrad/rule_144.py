import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if lr dtype is float, min value is zero, and update_slots are true, shape of accum must be greater or equal to 1 and less than 100 (Rule 144)

rule_144 = lambda s, v, n=False: (
    s.add(Not(If(And(And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), Select(v["arg1_range"], 0) == 0), v["arg2_value"] == True), And(Select(v["arg3_shape"], 0) >= 1, Select(v["arg3_shape"], 0) < 100), False)) if n else
          If(And(And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), Select(v["arg1_range"], 0) == 0), v["arg2_value"] == True), And(Select(v["arg3_shape"], 0) >= 1, Select(v["arg3_shape"], 0) < 100), False))
)

def rule_144_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 144
        rule_144(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_144(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)

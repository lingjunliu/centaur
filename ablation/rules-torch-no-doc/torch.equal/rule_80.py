import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check If min values and shape are the same they must be zero (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg3_value"] == Select(v["arg1_shape"], 0), v["arg4_value"] == Select(v["arg2_shape"], 0)), If((And((Select(v["arg1_range"], 0) == Select(v["arg2_range"], 0)), (v["arg3_value"] == v["arg4_value"]))), (And(v["arg3_value"] == 0, v["arg4_value"] == 0)), True))) if n else
          And(And(v["arg3_value"] == Select(v["arg1_shape"], 0), v["arg4_value"] == Select(v["arg2_shape"], 0)), If((And((Select(v["arg1_range"], 0) == Select(v["arg2_range"], 0)), (v["arg3_value"] == v["arg4_value"]))), (And(v["arg3_value"] == 0, v["arg4_value"] == 0)), True)))
)

def rule_80_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 80
        rule_80(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)

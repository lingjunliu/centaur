import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If a dense-dense multiplication, then don't use sparse matmul unless small (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_value"] == False), (v["arg2_value"] == False)), Or((Select(v["arg3_shape"], 0) * Select(v["arg3_shape"], 1) < 1000), (Select(v["arg4_shape"], 0) * Select(v["arg4_shape"], 1) < 1000)), True)) if n else
          If(And((v["arg1_value"] == False), (v["arg2_value"] == False)), Or((Select(v["arg3_shape"], 0) * Select(v["arg3_shape"], 1) < 1000), (Select(v["arg4_shape"], 0) * Select(v["arg4_shape"], 1) < 1000)), True))
)

def rule_99_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 99
        rule_99(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg4_shape': arg4['shape']}, neg)

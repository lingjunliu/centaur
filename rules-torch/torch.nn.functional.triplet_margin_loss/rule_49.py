import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If swap is True and reduction is NOT None, the total number of elements should match (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg4_value"] == True, v["arg5_value"] != 6), And((Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 0) * Select(v["arg2_shape"], 1)), (Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) == Select(v["arg3_shape"], 0) * Select(v["arg3_shape"], 1))), True)) if n else
          If(And(v["arg4_value"] == True, v["arg5_value"] != 6), And((Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) == Select(v["arg2_shape"], 0) * Select(v["arg2_shape"], 1)), (Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) == Select(v["arg3_shape"], 0) * Select(v["arg3_shape"], 1))), True))
)

def rule_49_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Bool('arg4_value')
        arg5_value = Int('arg5_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == list_of_string_values_torch.index(arg5))

        # Constraints for rule 49
        rule_49(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)

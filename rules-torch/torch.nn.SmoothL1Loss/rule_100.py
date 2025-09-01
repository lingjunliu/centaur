import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# beta must be non-negative, input and target tensors should not have Long data type, input and target tensors must have the same shape and Valid options for reduction parameter (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_value"] >= 0, v["arg2_dtype"] != 4), v["arg3_dtype"] != 4), v["arg2_ndim"] == v["arg3_ndim"]), And([Implies(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i), (Or(Or(v["arg4_value"] == 6, v["arg4_value"] == 7), v["arg4_value"] == 8)))) for i in range(6)]))) if n else
          And(And(And(And(v["arg1_value"] >= 0, v["arg2_dtype"] != 4), v["arg3_dtype"] != 4), v["arg2_ndim"] == v["arg3_ndim"]), And([Implies(i < (v["arg2_ndim"] - 1 + 1), And(Select(v["arg2_shape"], i) == Select(v["arg3_shape"], i), (Or(Or(v["arg4_value"] == 6, v["arg4_value"] == 7), v["arg4_value"] == 8)))) for i in range(6)])))
)

def rule_100_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_value = String('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == list_of_string_values_torch.index(arg4))

        # Constraints for rule 100
        rule_100(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)

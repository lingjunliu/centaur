import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If at least a tensor dimension is bigger than zero and swap is true margin has to be greater than zero (Rule 92)

rule_92 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And((Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), (Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]))), v["arg4_value"] == True), v["arg5_value"] > 0, False)) if n else
          If(And(And(And((Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)]))), (Or([And(i < (v["arg3_ndim"] - 1 + 1), Select(v["arg3_shape"], i) > 0) for i in range(6)]))), v["arg4_value"] == True), v["arg5_value"] > 0, False))
)

def rule_92_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        if not isinstance(arg5, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Bool('arg4_value')
        arg5_value = Real('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)

        # Constraints for rule 92
        rule_92(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_92(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)

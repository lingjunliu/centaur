import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If requires_grad is True, then if out is also specified, then n must be positive, d must be specified, dtype must be floating-point or complex, and the shape of out must be (n, (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg5_value"], v["arg3_ndim"] > 0), (And(And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0.0), (Or(Or(Or(v["arg4_value"] == 7, v["arg4_value"] == 8), v["arg4_value"] == 9), v["arg4_value"] == 10))), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) == v["arg1_value"])), True)) if n else
          If(And(v["arg5_value"], v["arg3_ndim"] > 0), (And(And(And(And(v["arg1_value"] > 0, v["arg2_value"] > 0.0), (Or(Or(Or(v["arg4_value"] == 7, v["arg4_value"] == 8), v["arg4_value"] == 9), v["arg4_value"] == 10))), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) == v["arg1_value"])), True))
)

def rule_47_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Int('arg4_value')
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))
        solver.add(arg5_value == arg5)

        # Constraints for rule 47
        rule_47(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# eps must be float, reduction must be string, shapes of input and target must be compatible for broadcasting and shapes of input and var must be compatible for broadcasting (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8)), v["arg1_value"] > 0), (And(And(v["arg3_ndim"] > 0, v["arg4_ndim"] > 0), And([Implies(i < (If(v["arg3_ndim"] >= v["arg4_ndim"], v["arg3_ndim"] - 1, v["arg4_ndim"] - 1) + 1), Or(Or(Or(Or(v["arg3_ndim"] - i - 1 < 0, v["arg4_ndim"] - i - 1 < 0), Select(v["arg3_shape"], v["arg3_ndim"] - i - 1) == 1), Select(v["arg4_shape"], v["arg4_ndim"] - i - 1) == 1), Select(v["arg3_shape"], v["arg3_ndim"] - i - 1) == Select(v["arg4_shape"], v["arg4_ndim"] - i - 1))) for i in range(6)])))), (Or(Or(v["arg3_ndim"] == v["arg5_ndim"], (And(v["arg3_ndim"] == v["arg5_ndim"] + 1, (Or(Select(v["arg5_shape"], v["arg5_ndim"] - 1) == 1, Select(v["arg5_shape"], v["arg5_ndim"] - 1) == Select(v["arg3_shape"], v["arg3_ndim"] - 1)))))), (v["arg5_ndim"] == 0))))) if n else
          And(And(And((Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8)), v["arg1_value"] > 0), (And(And(v["arg3_ndim"] > 0, v["arg4_ndim"] > 0), And([Implies(i < (If(v["arg3_ndim"] >= v["arg4_ndim"], v["arg3_ndim"] - 1, v["arg4_ndim"] - 1) + 1), Or(Or(Or(Or(v["arg3_ndim"] - i - 1 < 0, v["arg4_ndim"] - i - 1 < 0), Select(v["arg3_shape"], v["arg3_ndim"] - i - 1) == 1), Select(v["arg4_shape"], v["arg4_ndim"] - i - 1) == 1), Select(v["arg3_shape"], v["arg3_ndim"] - i - 1) == Select(v["arg4_shape"], v["arg4_ndim"] - i - 1))) for i in range(6)])))), (Or(Or(v["arg3_ndim"] == v["arg5_ndim"], (And(v["arg3_ndim"] == v["arg5_ndim"] + 1, (Or(Select(v["arg5_shape"], v["arg5_ndim"] - 1) == 1, Select(v["arg5_shape"], v["arg5_ndim"] - 1) == Select(v["arg3_shape"], v["arg3_ndim"] - 1)))))), (v["arg5_ndim"] == 0)))))
)

def rule_45_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = String('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_ndim = Int('arg5_ndim')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_string_values_torch.torch.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_ndim == arg4.ndim)
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg5_ndim == arg5.ndim)
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 45
        rule_45(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape, 'arg4_ndim': arg4_ndim, 'arg4_shape': arg4_shape, 'arg5_ndim': arg5_ndim, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape'], 'arg4_ndim': arg4['ndim'], 'arg4_shape': arg4['shape'], 'arg5_ndim': arg5['ndim'], 'arg5_shape': arg5['shape']}, neg)

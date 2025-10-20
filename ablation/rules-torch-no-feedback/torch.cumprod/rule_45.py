import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if out is specified, it should be a tensor and same size if dim is name (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10), v["arg2_value"] == 11), v["arg2_value"] == 12), v["arg2_value"] == 13), v["arg2_value"] == 14), v["arg2_value"] == 15), v["arg2_value"] == 16), v["arg2_value"] == 17), v["arg2_value"] == 18), v["arg2_value"] == 19), v["arg2_value"] == 20), v["arg2_value"] == 21), v["arg2_value"] == 22), v["arg2_value"] == 23), v["arg2_value"] == 24), v["arg2_value"] == 25), v["arg2_value"] == 26), v["arg2_value"] == 27), v["arg2_value"] == 28), v["arg2_value"] == 29), v["arg2_value"] == 20), And(v["arg1_ndim"] == v["arg3_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)) for i in range(6)])), True)) if n else
          If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10), v["arg2_value"] == 11), v["arg2_value"] == 12), v["arg2_value"] == 13), v["arg2_value"] == 14), v["arg2_value"] == 15), v["arg2_value"] == 16), v["arg2_value"] == 17), v["arg2_value"] == 18), v["arg2_value"] == 19), v["arg2_value"] == 20), v["arg2_value"] == 21), v["arg2_value"] == 22), v["arg2_value"] == 23), v["arg2_value"] == 24), v["arg2_value"] == 25), v["arg2_value"] == 26), v["arg2_value"] == 27), v["arg2_value"] == 28), v["arg2_value"] == 29), v["arg2_value"] == 20), And(v["arg1_ndim"] == v["arg3_ndim"], And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) == Select(v["arg3_shape"], i)) for i in range(6)])), True))
)

def rule_45_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 45
        rule_45(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_ndim': arg3['ndim']}, neg)

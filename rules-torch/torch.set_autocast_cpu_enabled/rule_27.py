import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If enabled, floating point tensor, requires positive dimension, else int with non negative dimension (Rule 27)

rule_27 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, (And(And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 10), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)])))), (And(And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) >= 0) for i in range(6)])))))) if n else
          If(v["arg1_value"] == True, (And(And(6 <= v["arg2_dtype"], v["arg2_dtype"] <= 10), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 0) for i in range(6)])))), (And(And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5), (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) >= 0) for i in range(6)]))))))
)

def rule_27_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 27
        rule_27(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_27(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)

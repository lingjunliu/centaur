import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If 'a' is a scalar, 'x' must also be a scalar or a tensor with compatible dtype and all non-negative values (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, (Or(v["arg2_ndim"] == 0, (And([Implies(i < (v["arg2_ndim"] - 1 + 1), And(And(Select(v["arg2_shape"], i) > 0, Select(v["arg2_range"], 0) >= 0), v["arg2_dtype"] == v["arg1_dtype"])) for i in range(6)])))), True)) if n else
          If(v["arg1_ndim"] == 0, (Or(v["arg2_ndim"] == 0, (And([Implies(i < (v["arg2_ndim"] - 1 + 1), And(And(Select(v["arg2_shape"], i) > 0, Select(v["arg2_range"], 0) >= 0), v["arg2_dtype"] == v["arg1_dtype"])) for i in range(6)])))), True))
)

def rule_39_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 39
        rule_39(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range']}, neg)

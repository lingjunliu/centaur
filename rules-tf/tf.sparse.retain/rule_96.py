import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Ensure sp_input is a tensor with valid shape and dimensions, and is not a scalar or primitive. (Rule 96)

rule_96 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] > 0, Or([And(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(And(Select(v["arg1_shape"], i) > 0, v["arg1_dtype"] != 11), v["arg1_dtype"] != 0), v["arg1_dtype"] != 1), v["arg1_dtype"] != 6), v["arg1_dtype"] != 12)) for i in range(6)]))) if n else
          And(v["arg1_ndim"] > 0, Or([And(i < (v["arg1_ndim"] - 1 + 1), And(And(And(And(And(Select(v["arg1_shape"], i) > 0, v["arg1_dtype"] != 11), v["arg1_dtype"] != 0), v["arg1_dtype"] != 1), v["arg1_dtype"] != 6), v["arg1_dtype"] != 12)) for i in range(6)])))
)

def rule_96_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 96
        rule_96(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)

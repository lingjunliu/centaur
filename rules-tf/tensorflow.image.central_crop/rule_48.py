import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Central fraction and image properties (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg2_value"] > 0.0, v["arg2_value"] <= 1.0), (Or((And(v["arg1_ndim"] == 3, And([Implies(i < (2 + 1), And(Select(v["arg1_shape"], i) > 0, (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6)))) for i in range(6)]))), (And(v["arg1_ndim"] == 4, And([Implies(i < (3 + 1), And(Select(v["arg1_shape"], i) > 0, (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6)))) for i in range(6)]))))))) if n else
          And(And(v["arg2_value"] > 0.0, v["arg2_value"] <= 1.0), (Or((And(v["arg1_ndim"] == 3, And([Implies(i < (2 + 1), And(Select(v["arg1_shape"], i) > 0, (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6)))) for i in range(6)]))), (And(v["arg1_ndim"] == 4, And([Implies(i < (3 + 1), And(Select(v["arg1_shape"], i) > 0, (Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 6)))) for i in range(6)])))))))
)

def rule_48_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 48
        rule_48(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Input must be a 0-dimensional tensor or a tensor where the product of its dimensions is 1 (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_ndim"] == 0, (Or([And(i < (v["arg1_ndim"] - 1 + 1), And((And([Implies(j < (v["arg1_ndim"] - 1 + 1), If(i == j, Select(v["arg1_shape"], i) > 0, Select(v["arg1_shape"], j) > 0)) for j in range(6)])), (And([Implies(k < (v["arg1_ndim"] - 1 + 1), (And([Implies(l < (v["arg1_ndim"] - 1 + 1), If(k != l, Select(v["arg1_shape"], k) * Select(v["arg1_shape"], l) == 1, False)) for l in range(6)]))) for k in range(6)])))) for i in range(6)])))) if n else
          Or(v["arg1_ndim"] == 0, (Or([And(i < (v["arg1_ndim"] - 1 + 1), And((And([Implies(j < (v["arg1_ndim"] - 1 + 1), If(i == j, Select(v["arg1_shape"], i) > 0, Select(v["arg1_shape"], j) > 0)) for j in range(6)])), (And([Implies(k < (v["arg1_ndim"] - 1 + 1), (And([Implies(l < (v["arg1_ndim"] - 1 + 1), If(k != l, Select(v["arg1_shape"], k) * Select(v["arg1_shape"], l) == 1, False)) for l in range(6)]))) for k in range(6)])))) for i in range(6)]))))
)

def rule_20_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 20
        rule_20(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)

import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# tensor is 0-dimensional OR if any dimension is not 1, there must exist another dimension that isn't 1 either - ensures only 1 element (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_ndim"] == 0, (And(v["arg1_ndim"] > 0, (If(Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) != 1) for i in range(6)]), Or([And(j < (v["arg1_ndim"] - 1 + 1), And(i != j, Select(v["arg1_shape"], j) != 1)) for j in range(6)]), True)))))) if n else
          Or(v["arg1_ndim"] == 0, (And(v["arg1_ndim"] > 0, (If(Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) != 1) for i in range(6)]), Or([And(j < (v["arg1_ndim"] - 1 + 1), And(i != j, Select(v["arg1_shape"], j) != 1)) for j in range(6)]), True))))))
)

def rule_11_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 11
        rule_11(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)

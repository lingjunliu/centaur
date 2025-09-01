import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The total number of elements in the tensor cannot be extremely large, and at least 1 dimension should not be zero. (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(And((Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), (If(v["arg1_ndim"] == 0, 1 < 100000000, If(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) < 100000000, If(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 100000000, If(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) < 100000000, If(v["arg1_ndim"] == 4, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) < 100000000, If(v["arg1_ndim"] == 5, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) * Select(v["arg1_shape"], 4) < 100000000, True))))))))) if n else
          And((Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])), (If(v["arg1_ndim"] == 0, 1 < 100000000, If(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) < 100000000, If(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 100000000, If(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) < 100000000, If(v["arg1_ndim"] == 4, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) < 100000000, If(v["arg1_ndim"] == 5, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) * Select(v["arg1_shape"], 4) < 100000000, True)))))))))
)

def rule_72_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 72
        rule_72(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)

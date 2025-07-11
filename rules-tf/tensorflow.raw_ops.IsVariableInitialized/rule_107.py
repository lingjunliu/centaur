import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check if tensor is a 3D tensor with each dimension's size less than 100 and greater than 0, shape[0] < shape[1] < shape[2] AND shape[0] + shape[1] + shape[2] < 200 (Rule 107)

rule_107 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == 3, And([Implies(i < (2 + 1), And(And(And(And(Select(v["arg1_shape"], i) < 100, Select(v["arg1_shape"], i) > 0), Select(v["arg1_shape"], 0) < Select(v["arg1_shape"], 1)), Select(v["arg1_shape"], 1) < Select(v["arg1_shape"], 2)), Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], 1) + Select(v["arg1_shape"], 2) < 200)) for i in range(6)]))) if n else
          And(v["arg1_ndim"] == 3, And([Implies(i < (2 + 1), And(And(And(And(Select(v["arg1_shape"], i) < 100, Select(v["arg1_shape"], i) > 0), Select(v["arg1_shape"], 0) < Select(v["arg1_shape"], 1)), Select(v["arg1_shape"], 1) < Select(v["arg1_shape"], 2)), Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], 1) + Select(v["arg1_shape"], 2) < 200)) for i in range(6)])))
)

def rule_107_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 107
        rule_107(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_107(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)

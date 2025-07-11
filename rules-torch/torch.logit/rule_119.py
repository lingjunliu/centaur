import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Values of the tensor have to be within the bounds of -1 to 1 for logit. Values cannot be NaN or infinity and all shapes has to be greater than 0. (Rule 119)

rule_119 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(Select(v["arg1_range"], 0) >= -1.0, Select(v["arg1_range"], 1) <= 1.0), Select(v["arg1_range"], 0) == Select(v["arg1_range"], 0)), Select(v["arg1_range"], 1) == Select(v["arg1_range"], 1)), Select(v["arg1_range"], 0) != (1.0 / 0.0) * (-1.0)), Select(v["arg1_range"], 1) != (1.0 / 0.0)), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)]))) if n else
          And(And(And(And(And(And(Select(v["arg1_range"], 0) >= -1.0, Select(v["arg1_range"], 1) <= 1.0), Select(v["arg1_range"], 0) == Select(v["arg1_range"], 0)), Select(v["arg1_range"], 1) == Select(v["arg1_range"], 1)), Select(v["arg1_range"], 0) != (1.0 / 0.0) * (-1.0)), Select(v["arg1_range"], 1) != (1.0 / 0.0)), And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 0) for i in range(6)])))
)

def rule_119_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 119
        rule_119(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_119(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)

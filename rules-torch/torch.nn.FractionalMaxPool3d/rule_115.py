import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Output_size and input should be tensors and should not result in negative dimensions. (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 5, And(And((Select(v["arg1_shape"], 2) * Select(v["arg2_values"], 0)) >= 0, (Select(v["arg1_shape"], 3) * Select(v["arg2_values"], 1)) >= 0), (Select(v["arg1_shape"], 4) * Select(v["arg2_values"], 2)) >= 0), False)) if n else
          If(v["arg1_ndim"] == 5, And(And((Select(v["arg1_shape"], 2) * Select(v["arg2_values"], 0)) >= 0, (Select(v["arg1_shape"], 3) * Select(v["arg2_values"], 1)) >= 0), (Select(v["arg1_shape"], 4) * Select(v["arg2_values"], 2)) >= 0), False))
)

def rule_115_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 115
        rule_115(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_values': arg2['values']}, neg)

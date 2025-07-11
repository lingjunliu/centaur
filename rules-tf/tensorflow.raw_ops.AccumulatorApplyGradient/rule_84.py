import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the gradient is a 3D tensor, its shape dimensions must form a Pythagorean triple, i.e a*a + b*b = c*c where a,b, and c represent dimensions of the shape array (Rule 84)

rule_84 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 1) == Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 2), False)) if n else
          If(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 1) == Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 2), False))
)

def rule_84_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 84
        rule_84(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_84(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)

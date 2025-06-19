import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If a tensor is 2D, and the first element is 5, then the second must be greater than 5 (Rule 366)

rule_366 = lambda s, v: (
    s.add(If(And(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) == 5), Select(v["arg1_shape"], 1) > 5, False))
)

def rule_366_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 366
        rule_366(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_366(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']})

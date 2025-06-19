import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If there are only 2 dimensions in the tensor, and one of these dimension equals 1, then all shapes of that tensor must be equal to each other (Rule 415)

rule_415 = lambda s, v: (
    s.add(If(And(v["arg1_ndim"] == 2, (Or(Select(v["arg1_shape"], 0) == 1, Select(v["arg1_shape"], 1) == 1))), Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], 1), False))
)

def rule_415_func(arg1, solver=None):
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

        # Constraints for rule 415
        rule_415(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_415(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']})

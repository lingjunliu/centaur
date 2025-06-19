import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If shape of 0 dimension of the tensor is not 1, all elements of the tensor must be positive number (Rule 446)

rule_446 = lambda s, v: (
    s.add(If(Select(v["arg1_shape"], 0) != 1, Select(v["arg1_range"], 0) >= 0, False))
)

def rule_446_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 446
        rule_446(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_446(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range']})

import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the max value is larger than 10, all dimensions must be same value (Rule 179)

rule_179 = lambda s, v: (
    s.add(If(Select(v["arg1_range"], 1) > 10, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], i)) for i in range(6)]), True))
)

def rule_179_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
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

        # Constraints for rule 179
        rule_179(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_179(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape']})

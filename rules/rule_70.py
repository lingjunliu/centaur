import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If number of dimensions is non-zero and even, the max value must be bigger than 5 (Rule 70)

rule_70 = lambda s, v: (
    s.add(If(And(v["arg1_ndim"] > 0, v["arg1_ndim"] * 0.5 == v["arg1_ndim"] / 2), Select(v["arg1_range"], 1) > 5, True))
)

def rule_70_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 70
        rule_70(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range']})

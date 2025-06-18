import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# The number of dimensions of a tensor must be a multiple of 2 (Rule 62)

rule_62 = lambda s, v: (
    s.add(v["arg1_ndim"] * 0.5 == v["arg1_ndim"] / 2)
)

def rule_62_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 62
        rule_62(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_ndim': arg1['ndim']})

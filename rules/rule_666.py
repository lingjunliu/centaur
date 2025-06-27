import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# The tensor should not be a scalar (0 dimension (Rule 666)

rule_666 = lambda s, v, n=False: (
    s.add(Not(v["arg1_ndim"] > 0) if n else
          v["arg1_ndim"] > 0)
)

def rule_666_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 666
        rule_666(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_666(solver, {'arg1_ndim': arg1['ndim']}, neg)

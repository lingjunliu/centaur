import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# v_1 must be a valid dimension of v_2, specifically, it cannot be the last one (Rule 331)

rule_331 = lambda s, v, n=False: (
    s.add(Not(And(-1 * v["arg2_ndim"] <= v["arg1_value"], v["arg1_value"] <= v["arg2_ndim"] - 2)) if n else
          And(-1 * v["arg2_ndim"] <= v["arg1_value"], v["arg1_value"] <= v["arg2_ndim"] - 2))
)

def rule_331_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 331
        rule_331(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_331(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)

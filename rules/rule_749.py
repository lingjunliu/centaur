import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If the tensor v_1 is 2 dimensional, then the minimum value in dimension 0 should be smaller than the maximum value in dimension 1. (Rule 749)

rule_749 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 2, Select(v["arg1_range"], 0) < Select(v["arg1_range"], 1), False)) if n else
          If(v["arg1_ndim"] == 2, Select(v["arg1_range"], 0) < Select(v["arg1_range"], 1), False))
)

def rule_749_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 749
        rule_749(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_749(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)

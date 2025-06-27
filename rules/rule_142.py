import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if the max value of the tensor <10, then dimension should be smaller than or equal to 5 (Rule 142)

rule_142 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_range"], 1) < 10, v["arg1_ndim"] <= 5, False)) if n else
          If(Select(v["arg1_range"], 1) < 10, v["arg1_ndim"] <= 5, False))
)

def rule_142_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 142
        rule_142(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_142(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)

import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If input tensor v_1 has more than 2 dimensions, maximum element must be greater than the number 10 (Rule 767)

rule_767 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 2, Select(v["arg1_range"], 1) > 10, False)) if n else
          If(v["arg1_ndim"] > 2, Select(v["arg1_range"], 1) > 10, False))
)

def rule_767_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 767
        rule_767(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_767(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)

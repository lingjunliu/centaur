import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If the dimension of the tensor is greater than 3, then max should also be greater than min * 2 (Rule 87)

rule_87 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 3, Select(v["arg1_range"], 1) > Select(v["arg1_range"], 0) * 2, False)) if n else
          If(v["arg1_ndim"] > 3, Select(v["arg1_range"], 1) > Select(v["arg1_range"], 0) * 2, False))
)

def rule_87_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 87
        rule_87(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_87(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)

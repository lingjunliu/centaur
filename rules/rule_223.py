import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If number of dimensions of the tensor is odd, the min must be > 0 and max < 100, otherwise, the min must be < 0 and max > -100. (Rule 223)

rule_223 = lambda s, v: (
    s.add(If(v["arg1_ndim"] / 2 == (v["arg1_ndim"] / 2), (And(Select(v["arg1_range"], 0) < 0, Select(v["arg1_range"], 1) > -100)), (And(Select(v["arg1_range"], 0) > 0, Select(v["arg1_range"], 1) < 100))))
)

def rule_223_func(arg1, solver=None):
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

        # Constraints for rule 223
        rule_223(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_223(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']})

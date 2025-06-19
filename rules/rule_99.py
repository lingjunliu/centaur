import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If tensor v_1's dtype is complex, then its min must be strictly less than 0, or its max strictly greater than zero (Rule 99)

rule_99 = lambda s, v: (
    s.add(If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), Or(Select(v["arg1_range"], 0) < 0, Select(v["arg1_range"], 1) > 0), False))
)

def rule_99_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 99
        rule_99(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']})

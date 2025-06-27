import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If a tensor's dtype is not boolean, then min must be non negative. (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] != 0, Select(v["arg1_range"], 0) >= 0, False)) if n else
          If(v["arg1_dtype"] != 0, Select(v["arg1_range"], 0) >= 0, False))
)

def rule_45_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 45
        rule_45(solver, {'arg1_range': arg1_range, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_range': arg1['range'], 'arg1_dtype_': arg1['dtype_']}, neg)

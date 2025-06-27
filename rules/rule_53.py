import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If dtype is Complex64 or Complex128, then max should be less than or equal to 10000 (Rule 53)

rule_53 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), Select(v["arg1_range"], 1) <= 10000, False)) if n else
          If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), Select(v["arg1_range"], 1) <= 10000, False))
)

def rule_53_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 53
        rule_53(solver, {'arg1_range': arg1_range, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_53(solver, {'arg1_range': arg1['range'], 'arg1_dtype_': arg1['dtype_']}, neg)

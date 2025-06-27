import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If a tensor is boolean or its number of dimensions is 0 then minimum of this tensor will not be greater than zero (Rule 791)

rule_791 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 0, v["arg1_ndim"] == 0), Select(v["arg1_range"], 0) <= 0, False)) if n else
          If(Or(v["arg1_dtype"] == 0, v["arg1_ndim"] == 0), Select(v["arg1_range"], 0) <= 0, False))
)

def rule_791_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 791
        rule_791(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_791(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_']}, neg)

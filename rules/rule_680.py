import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If tensor v_1 is 2D, then its dtype must be different from bool (Rule 680)

rule_680 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 2, v["arg1_dtype"] != 0, False)) if n else
          If(v["arg1_ndim"] == 2, v["arg1_dtype"] != 0, False))
)

def rule_680_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 680
        rule_680(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_680(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_']}, neg)

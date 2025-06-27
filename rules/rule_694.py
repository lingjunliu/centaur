import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# The tensor should be an integer type and ndim is less than 3 or the tensor should be a float type and ndim is larger than 1 (Rule 694)

rule_694 = lambda s, v, n=False: (
    s.add(Not(Or((And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), v["arg1_ndim"] < 3)), (And(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), v["arg1_ndim"] > 1)))) if n else
          Or((And(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), v["arg1_ndim"] < 3)), (And(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 8), v["arg1_ndim"] > 1))))
)

def rule_694_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 694
        rule_694(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_694(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_']}, neg)

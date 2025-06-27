import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# For a tensor v_1 if its number of dimensions is less than 2, its dtype should be float (Rule 701)

rule_701 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] < 2, Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), False)) if n else
          If(v["arg1_ndim"] < 2, Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), False))
)

def rule_701_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 701
        rule_701(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_701(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_']}, neg)

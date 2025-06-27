import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If the dtype of input tensor v_1 is string then number of dimensions should be 1 (Rule 731)

rule_731 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, v["arg1_ndim"] == 1, False)) if n else
          If(v["arg1_dtype"] == 11, v["arg1_ndim"] == 1, False))
)

def rule_731_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 731
        rule_731(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_731(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_']}, neg)

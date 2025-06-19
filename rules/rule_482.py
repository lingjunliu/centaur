import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if tensor v1's data type equals to a string, then tensor v1's number of dimensions must be greater than one (Rule 482)

rule_482 = lambda s, v: (
    s.add(If(v["arg1_dtype"] == 11, v["arg1_ndim"] > 1, False))
)

def rule_482_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 482
        rule_482(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_482(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']})

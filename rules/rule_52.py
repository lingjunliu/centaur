import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If ndim is 2, the tensor should not be a boolean type (Rule 52)

rule_52 = lambda s, v: (
    s.add(If(v["arg1_ndim"] == 2, v["arg1_dtype"] != 0, True))
)

def rule_52_func(arg1, solver=None):
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

        # Constraints for rule 52
        rule_52(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype']})

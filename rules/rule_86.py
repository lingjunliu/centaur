import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If 0 < ndim <= 2, then dtype cannot be complex. (Rule 86)

rule_86 = lambda s, v: (
    s.add(If(And((v["arg1_ndim"] > 0), (v["arg1_ndim"] <= 2)), And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10), True))
)

def rule_86_func(arg1, solver=None):
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

        # Constraints for rule 86
        rule_86(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_86(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype']})

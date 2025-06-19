import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the number of dimensions is 1, then the dtype of the tensor should be an integer (Rule 278)

rule_278 = lambda s, v: (
    s.add(If(v["arg1_ndim"] == 1, And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), False))
)

def rule_278_func(arg1, solver=None):
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

        # Constraints for rule 278
        rule_278(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_278(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']})

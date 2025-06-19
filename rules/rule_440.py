import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if tensors v1 and v2 are tensors and the dimension of v1 equals to dimension of v2 plus 1 then the maximum element must be greater than 0 (Rule 440)

rule_440 = lambda s, v: (
    s.add(If(v["arg1_ndim"] == v["arg2_ndim"] + 1, Select(v["arg1_range"], 1) > 0, False))
)

def rule_440_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 440
        rule_440(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_440(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']})

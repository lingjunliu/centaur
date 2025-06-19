import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If the ndim of tensor v_1 is equal to 2, then the multiplication of its shape at dimension 0 and shape at dimension 1 should be greater than 10 (Rule 37)

rule_37 = lambda s, v: (
    s.add(If(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) > 10, False))
)

def rule_37_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 37
        rule_37(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']})

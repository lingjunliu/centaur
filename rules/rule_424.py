import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if max value of a tensor is greater than zero, then there must exist at least one dimension i such that shape(v1,i (Rule 424)

rule_424 = lambda s, v: (
    s.add(If(Select(v["arg1_range"], 1) > 0, Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > Select(v["arg1_range"], 0)) for i in range(6)]), False))
)

def rule_424_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 424
        rule_424(solver, {'arg1_shape': arg1_shape, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_424(solver, {'arg1_shape': arg1['shape'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']})

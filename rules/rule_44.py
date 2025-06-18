import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if ndim is 0 then min and max should be 0 (Rule 44)

rule_44 = lambda s, v: (
    s.add(If(v["arg1_ndim"] == 0, And(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 1) == 0), True))
)

def rule_44_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 44
        rule_44(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range']})

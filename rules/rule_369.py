import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If a tensor has a minimum and maximum value equal to zero, then the tensor must either have a bool or integer dtype. (Rule 369)

rule_369 = lambda s, v: (
    s.add(If(And(Select(v["arg1_range"], 0) == 0, Select(v["arg1_range"], 1) == 0), Or(v["arg1_dtype"] == 0, (And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5))), False))
)

def rule_369_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 369
        rule_369(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_369(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']})

import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If tensor value is string or integer, string must be in range [0,256] (Rule 195)

rule_195 = lambda s, v: (
    s.add(If(Or((v["arg1_dtype"] == 11), (And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5))), And(Select(v["arg1_range"], 1) <= 256, Select(v["arg1_range"], 0) >= 0), True))
)

def rule_195_func(arg1, solver=None):
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

        # Constraints for rule 195
        rule_195(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_195(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']})

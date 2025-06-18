import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If tensor has a boolean dtype, then both min and max should be either true or false (Rule 90)

rule_90 = lambda s, v: (
    s.add(If(v["arg1_dtype"] == 0, Or((And(Select(v["arg1_range"], 0) == True, Select(v["arg1_range"], 1) == True)), (And(Select(v["arg1_range"], 0) == False, Select(v["arg1_range"], 1) == False))), True))
)

def rule_90_func(arg1, solver=None):
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

        # Constraints for rule 90
        rule_90(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_90(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']})

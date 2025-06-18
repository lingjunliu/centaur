import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If dtype of the tensor is not integer 1-5 and all values in tensor are boolean then max must be greater than -1. (Rule 113)

rule_113 = lambda s, v: (
    s.add(If(And((Or(v["arg1_dtype"] < 1, v["arg1_dtype"] > 5)), (Or(Select(v["arg1_range"], 0) == True, Select(v["arg1_range"], 0) == False))), Select(v["arg1_range"], 1) > -1, True))
)

def rule_113_func(arg1, solver=None):
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

        # Constraints for rule 113
        rule_113(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_113(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']})

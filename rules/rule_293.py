import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if the dtype of the tensor v_1 is smaller than the maximum value of tensor, then the minimum value of v_1 has to be larger than -100 (Rule 293)

rule_293 = lambda s, v: (
    s.add(If(v["arg1_dtype"] < Select(v["arg1_range"], 1), Select(v["arg1_range"], 0) > -100, False))
)

def rule_293_func(arg1, solver=None):
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

        # Constraints for rule 293
        rule_293(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_293(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']})

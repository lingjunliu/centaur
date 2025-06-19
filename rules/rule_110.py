import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if the data type of tensor v_1 is in range 0–10, then the minimum value of v_1 must not be greater than the maximum value of v_1 (Rule 110)

rule_110 = lambda s, v: (
    s.add(If(And(0 <= v["arg1_dtype"], v["arg1_dtype"] <= 10), Select(v["arg1_range"], 0) <= Select(v["arg1_range"], 1), False))
)

def rule_110_func(arg1, solver=None):
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

        # Constraints for rule 110
        rule_110(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_110(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype']})

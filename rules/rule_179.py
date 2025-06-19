import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if bool v_1 is true, then tensor v_2 must be 2 dimensional, and must not have type as string nor its max should be smaller or equal than 0 (Rule 179)

rule_179 = lambda s, v: (
    s.add(If(v["arg1_value"], And(And(v["arg2_ndim"] == 2, v["arg2_dtype"] != 11), Select(v["arg2_range"], 1) > 0), False))
)

def rule_179_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 179
        rule_179(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_179(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']})

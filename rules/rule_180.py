import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If string v_1 is equal to "bool", then tensor v_2's data type must be equal to boolean or the minimum and maximum values of tensor v_2 should not be equal (Rule 180)

rule_180 = lambda s, v: (
    s.add(If(v["arg1_value"] == "bool", Or(v["arg2_dtype"] == 0, Select(v["arg2_range"], 0) != Select(v["arg2_range"], 1)), False))
)

def rule_180_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 180
        rule_180(solver, {'arg1_value': arg1_value, 'arg2_range': arg2_range, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_180(solver, {'arg1_value': arg1['value'], 'arg2_range': arg2['range'], 'arg2_dtype': arg2['dtype']})

import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If dtype is int, tensor must have same min and max value if string == same (Rule 158)

rule_158 = lambda s, v: (
    s.add(If(And(And((v["arg1_dtype"] >= 1), (v["arg1_dtype"] <= 5)), (v["arg2_value"] == "same")), (Select(v["arg1_range"], 1) == Select(v["arg1_range"], 0)), True))
)

def rule_158_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 158
        rule_158(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_158(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']})

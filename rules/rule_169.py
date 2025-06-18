import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If String has value "less" and float has value larger than 0, than value from min of Tensor must be great than floting parameter multply by 2 (Rule 169)

rule_169 = lambda s, v: (
    s.add(If(And((v["arg2_value"] == "less"), (v["arg3_value"] > 0)), Select(v["arg1_range"], 0) > (v["arg3_value"] * 2), True))
)

def rule_169_func(arg1, arg2, arg3, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, str)):
            return False
        if not (isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_value = Real('arg3_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 169
        rule_169(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_169(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']})

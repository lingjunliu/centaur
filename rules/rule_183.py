import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# String variable and the tensor maximum value must not be same as True or False (Rule 183)

rule_183 = lambda s, v: (
    s.add(And(And(And((v["arg2_value"] != "true"), (v["arg2_value"] != "false")), (Select(v["arg1_range"], 1) != "true")), (Select(v["arg1_range"], 1) != "false")))
)

def rule_183_func(arg1, arg2, solver=None):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == arg2)

        # Constraints for rule 183
        rule_183(solver, {'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_183(solver, {'arg1_range': arg1['range'], 'arg2_value': arg2['value']})

import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# If a string is not empty then length of string should be smaller than 5 (Rule 38)

rule_38 = lambda s, v: (
    s.add(If(v["arg1_value"] != "", v["arg1_value"] < "aaaaa", True))
)

def rule_38_func(arg1, solver=None):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value']})

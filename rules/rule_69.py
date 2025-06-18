import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# String should be contained in a given list of strings (Rule 69)

rule_69 = lambda s, v: (
    s.add(Or(Or(v["arg1_value"] == "a", v["arg1_value"] == "b"), v["arg1_value"] == "c"))
)

def rule_69_func(arg1, solver=None):
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

        # Constraints for rule 69
        rule_69(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_value': arg1['value']})

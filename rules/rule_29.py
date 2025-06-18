import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# String should contain a specific substring (Rule 29)

rule_29 = lambda s, v: (
    s.add(Or([And(i < (0 + 1), v["arg1_value"] == "hello") for i in range(6)]))
)

def rule_29_func(arg1, solver=None):
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

        # Constraints for rule 29
        rule_29(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_value': arg1['value']})

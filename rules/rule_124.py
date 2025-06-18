import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# A string variable should not be equal to any function names (Rule 124)

rule_124 = lambda s, v: (
    s.add(And(And(And(And(v["arg1_value"] != "ndim", v["arg1_value"] != "shape"), v["arg1_value"] != "dtype"), v["arg1_value"] != "min"), v["arg1_value"] != "max"))
)

def rule_124_func(arg1, solver=None):
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

        # Constraints for rule 124
        rule_124(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_124(solver, {'arg1_value': arg1['value']})

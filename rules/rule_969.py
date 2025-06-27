import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If pin_memory is set to True, a pin_memory allocator must be available (Rule 969)

rule_969 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] == False) if n else
          v["arg1_value"] == False)
)

def rule_969_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 969
        rule_969(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_969(solver, {'arg1_value': arg1['value']}, neg)

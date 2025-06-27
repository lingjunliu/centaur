import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If v_1's dtype is float, then v_2 should be greater than 0. (Rule 329)

rule_329 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_dtype"] == 8, v["arg2_value"] > 0)) if n else
          And(v["arg1_dtype"] == 8, v["arg2_value"] > 0))
)

def rule_329_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg2_value == arg2)

        # Constraints for rule 329
        rule_329(solver, {'arg1_dtype_': arg1_dtype_, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_329(solver, {'arg1_dtype_': arg1['dtype_'], 'arg2_value': arg2['value']}, neg)

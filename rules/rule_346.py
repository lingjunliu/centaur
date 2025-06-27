import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If the dtype of v_1 is between 1 and 5, then v_2 must be between 0 and 1 (Rule 346)

rule_346 = lambda s, v, n=False: (
    s.add(Not(If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And(0 <= v["arg2_value"], v["arg2_value"] <= 1), False)) if n else
          If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And(0 <= v["arg2_value"], v["arg2_value"] <= 1), False))
)

def rule_346_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 346
        rule_346(solver, {'arg1_dtype_': arg1_dtype_, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_346(solver, {'arg1_dtype_': arg1['dtype_'], 'arg2_value': arg2['value']}, neg)

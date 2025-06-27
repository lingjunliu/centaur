import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If the string is 'tanh' then the dtype of tensor v_1 should be float 32 or float 64 (Rule 708)

rule_708 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 11, Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), False)) if n else
          If(v["arg2_value"] == 11, Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), False))
)

def rule_708_func(arg1, arg2, solver=None, neg=False):
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
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 708
        rule_708(solver, {'arg1_dtype_': arg1_dtype_, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_708(solver, {'arg1_dtype_': arg1['dtype_'], 'arg2_value': arg2['value']}, neg)

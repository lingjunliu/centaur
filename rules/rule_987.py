import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# The output tensor's dtype must be Int(int32 (Rule 987)

rule_987 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, v["arg1_dtype"] == 3, v["arg1_dtype"] == 4)) if n else
          If(v["arg2_value"] == True, v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))
)

def rule_987_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg2_value == arg2)

        # Constraints for rule 987
        rule_987(solver, {'arg1_dtype_': arg1_dtype_, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_987(solver, {'arg1_dtype_': arg1['dtype_'], 'arg2_value': arg2['value']}, neg)

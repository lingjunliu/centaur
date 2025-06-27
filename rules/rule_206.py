import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if tensor's data type is not string type and string's value is tanh, then the min must be greater than -2 (Rule 206)

rule_206 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] != 11, v["arg2_value"] == 11), Select(v["arg1_range"], 0) > -2, False)) if n else
          If(And(v["arg1_dtype"] != 11, v["arg2_value"] == 11), Select(v["arg1_range"], 0) > -2, False))
)

def rule_206_func(arg1, arg2, solver=None, neg=False):
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
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 206
        rule_206(solver, {'arg1_range': arg1_range, 'arg1_dtype_': arg1_dtype_, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_206(solver, {'arg1_range': arg1['range'], 'arg1_dtype_': arg1['dtype_'], 'arg2_value': arg2['value']}, neg)

import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if v_1 is tanh, then v_2 tensor must be float16 or float32 or float64 (Rule 348)

rule_348 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 11, Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), False)) if n else
          If(v["arg1_value"] == 11, Or(Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), v["arg2_dtype"] == 9), False))
)

def rule_348_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values.index(arg1))

        # Constraints for rule 348
        rule_348(solver, {'arg1_value': arg1_value, 'arg2_dtype_': arg2_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_348(solver, {'arg1_value': arg1['value'], 'arg2_dtype_': arg2['dtype_']}, neg)

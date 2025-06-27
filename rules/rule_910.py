import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if the string is constant and number of dimensions equals to one or the number of dimension is zero, then type should not be boolean (Rule 910)

rule_910 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == 10, (Or(v["arg1_ndim"] == 0, v["arg1_ndim"] == 1))), v["arg1_dtype"] != 0, False)) if n else
          If(And(v["arg2_value"] == 10, (Or(v["arg1_ndim"] == 0, v["arg1_ndim"] == 1))), v["arg1_dtype"] != 0, False))
)

def rule_910_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 910
        rule_910(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_910(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_'], 'arg2_value': arg2['value']}, neg)

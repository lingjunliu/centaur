import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if dtype is not str and the number of dimensions is greater than 2 and min is larger than 0 then string is not allowed to be constant (Rule 196)

rule_196 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_dtype"] != 11, v["arg1_ndim"] > 2), Select(v["arg1_range"], 0) > 0), v["arg2_value"] != 10, False)) if n else
          If(And(And(v["arg1_dtype"] != 11, v["arg1_ndim"] > 2), Select(v["arg1_range"], 0) > 0), v["arg2_value"] != 10, False))
)

def rule_196_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 196
        rule_196(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_196(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_'], 'arg2_value': arg2['value']}, neg)

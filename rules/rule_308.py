import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If tensor's max = 10, and its dtype is not string nor boolean and ndim is larger or equal to 2, then its string must be "sum" or dimension must be equal to 2. (Rule 308)

rule_308 = lambda s, v, n=False: (
    s.add(Not(If(And(And((Select(v["arg1_range"], 1) == 10), (And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11))), (v["arg1_ndim"] >= 2)), Or((v["arg2_value"] == 8), (v["arg1_ndim"] == 2)), False)) if n else
          If(And(And((Select(v["arg1_range"], 1) == 10), (And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 11))), (v["arg1_ndim"] >= 2)), Or((v["arg2_value"] == 8), (v["arg1_ndim"] == 2)), False))
)

def rule_308_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 308
        rule_308(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype_': arg1_dtype_, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_308(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype_': arg1['dtype_'], 'arg2_value': arg2['value']}, neg)

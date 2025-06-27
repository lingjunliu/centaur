import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If the dtype is bool, max value == min value and greater than max value of the float tensor (Rule 864)

rule_864 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_dtype"] == 0), And(Select(v["arg1_range"], 1) == Select(v["arg1_range"], 0), Select(v["arg1_range"], 1) > Select(v["arg2_range"], 1)), False)) if n else
          If((v["arg1_dtype"] == 0), And(Select(v["arg1_range"], 1) == Select(v["arg1_range"], 0), Select(v["arg1_range"], 1) > Select(v["arg2_range"], 1)), False))
)

def rule_864_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 864
        rule_864(solver, {'arg1_range': arg1_range, 'arg1_dtype_': arg1_dtype_, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_864(solver, {'arg1_range': arg1['range'], 'arg1_dtype_': arg1['dtype_'], 'arg2_range': arg2['range']}, neg)

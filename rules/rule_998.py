import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If out is provided, it should be a tensor and must be the same type as the other tensors (Rule 998)

rule_998 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_dtype"] == v["arg2_dtype"], v["arg2_dtype"] == v["arg3_dtype"])) if n else
          And(v["arg1_dtype"] == v["arg2_dtype"], v["arg2_dtype"] == v["arg3_dtype"]))
)

def rule_998_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False
        if not (isinstance(arg3, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 998
        rule_998(solver, {'arg1_dtype_': arg1_dtype_, 'arg2_dtype_': arg2_dtype_, 'arg3_dtype_': arg3_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_998(solver, {'arg1_dtype_': arg1['dtype_'], 'arg2_dtype_': arg2['dtype_'], 'arg3_dtype_': arg3['dtype_']}, neg)

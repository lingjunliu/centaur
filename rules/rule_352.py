import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if v_1 tensor's dtype is integer, then v_2 tensor's dtype should also be integer (Rule 352)

rule_352 = lambda s, v, n=False: (
    s.add(Not(If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5), False)) if n else
          If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 5), False))
)

def rule_352_func(arg1, arg2, solver=None, neg=False):
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

        # Value assignments

        # Constraints for rule 352
        rule_352(solver, {'arg1_dtype_': arg1_dtype_, 'arg2_dtype_': arg2_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_352(solver, {'arg1_dtype_': arg1['dtype_'], 'arg2_dtype_': arg2['dtype_']}, neg)

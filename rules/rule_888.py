import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# if dtype is str, then minimum value of v_1 is greater than shape value on 0th dimension (Rule 888)

rule_888 = lambda s, v, n=False: (
    s.add(Not(If((v["arg1_dtype"] == 11), Select(v["arg1_range"], 0) > Select(v["arg1_shape"], 0), False)) if n else
          If((v["arg1_dtype"] == 11), Select(v["arg1_range"], 0) > Select(v["arg1_shape"], 0), False))
)

def rule_888_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 888
        rule_888(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_888(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape'], 'arg1_dtype_': arg1['dtype_']}, neg)

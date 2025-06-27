import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If tensor data type is string, then its length must smaller than 50 (Rule 896)

rule_896 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, Select(v["arg1_shape"], 0) < 50, False)) if n else
          If(v["arg1_dtype"] == 11, Select(v["arg1_shape"], 0) < 50, False))
)

def rule_896_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 896
        rule_896(solver, {'arg1_shape': arg1_shape, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_896(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype_': arg1['dtype_']}, neg)

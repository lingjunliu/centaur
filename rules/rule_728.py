import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# If the shape of the input tensor's first dimension is less than 5, then the dtype must be bool (Rule 728)

rule_728 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) < 5, v["arg1_dtype"] == 0, False)) if n else
          If(Select(v["arg1_shape"], 0) < 5, v["arg1_dtype"] == 0, False))
)

def rule_728_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 728
        rule_728(solver, {'arg1_shape': arg1_shape, 'arg1_dtype_': arg1_dtype_})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_728(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype_': arg1['dtype_']}, neg)

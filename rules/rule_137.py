import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes
from z3 import *

# if there is a boolean and a tensor, and the boolean is true all the dimensions of tensor should be greater than 5. (Rule 137)

rule_137 = lambda s, v: (
    s.add(If(v["arg1_value"] == True, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) > 5) for i in range(6)]), True))
)

def rule_137_func(arg1, arg2, solver=None):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, np.ndarray)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 137
        rule_137(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_137(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']})
